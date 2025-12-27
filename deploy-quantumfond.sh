#!/bin/bash
# QuantumFond Deployment Script
# Usage: ./deploy-quantumfond.sh [environment]
# Example: ./deploy-quantumfond.sh production

set -e

ENVIRONMENT=${1:-production}
BACKEND_DIR="/opt/quantumfond/backend"
FRONTEND_DIR="/var/www/quantumfond/frontend"
BACKUP_DIR="/opt/quantumfond/backups"

echo "🚀 QuantumFond Deployment Script"
echo "Environment: $ENVIRONMENT"
echo "─────────────────────────────────"

# Create backup directory
mkdir -p $BACKUP_DIR

# Function to backup database
backup_database() {
    echo "📦 Backing up database..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    pg_dump -U quantumfond quantumdb > "$BACKUP_DIR/db_backup_$TIMESTAMP.sql"
    echo "✓ Database backed up to $BACKUP_DIR/db_backup_$TIMESTAMP.sql"
}

# Function to deploy backend
deploy_backend() {
    echo ""
    echo "🔧 Deploying Backend..."
    
    # Create backend directory if it doesn't exist
    sudo mkdir -p $BACKEND_DIR
    
    # Copy backend files
    echo "→ Copying backend files..."
    sudo cp -r quantumfond_backend/* $BACKEND_DIR/
    
    # Create virtual environment
    if [ ! -d "$BACKEND_DIR/venv" ]; then
        echo "→ Creating virtual environment..."
        cd $BACKEND_DIR
        sudo python3 -m venv venv
    fi
    
    # Install dependencies
    echo "→ Installing dependencies..."
    cd $BACKEND_DIR
    sudo $BACKEND_DIR/venv/bin/pip install -r requirements.txt
    
    # Copy environment file
    if [ ! -f "$BACKEND_DIR/.env" ]; then
        echo "→ Creating .env file..."
        sudo cp .env.quantumfond $BACKEND_DIR/.env
        echo "⚠️  Please configure $BACKEND_DIR/.env with production settings!"
    fi
    
    # Set permissions
    sudo chown -R quantumfond:quantumfond $BACKEND_DIR
    sudo chmod -R 755 $BACKEND_DIR
    
    # Initialize database
    echo "→ Initializing database..."
    cd $BACKEND_DIR
    sudo -u quantumfond $BACKEND_DIR/venv/bin/python -c "from db.connection import init_db; init_db()"
    
    # Install systemd service
    echo "→ Installing systemd service..."
    sudo cp quantumfond-backend.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable quantumfond-backend
    
    # Restart service
    echo "→ Restarting backend service..."
    sudo systemctl restart quantumfond-backend
    
    # Check status
    sleep 3
    if sudo systemctl is-active --quiet quantumfond-backend; then
        echo "✓ Backend deployed successfully!"
    else
        echo "✗ Backend deployment failed. Check logs: journalctl -u quantumfond-backend"
        exit 1
    fi
}

# Function to deploy frontend
deploy_frontend() {
    echo ""
    echo "🎨 Deploying Frontend..."
    
    # Build frontend
    echo "→ Building frontend..."
    cd quantumfond_frontend
    npm install
    npm run build
    
    # Create frontend directory
    sudo mkdir -p $FRONTEND_DIR
    
    # Copy built files
    echo "→ Copying frontend files..."
    sudo rm -rf $FRONTEND_DIR/*
    sudo cp -r dist/* $FRONTEND_DIR/
    
    # Set permissions
    sudo chown -R www-data:www-data $FRONTEND_DIR
    sudo chmod -R 755 $FRONTEND_DIR
    
    echo "✓ Frontend deployed successfully!"
}

# Function to configure nginx
configure_nginx() {
    echo ""
    echo "🌐 Configuring Nginx..."
    
    # Copy nginx configuration
    sudo cp nginx-quantumfond.conf /etc/nginx/sites-available/quantumfond
    
    # Enable site
    if [ ! -L /etc/nginx/sites-enabled/quantumfond ]; then
        sudo ln -s /etc/nginx/sites-available/quantumfond /etc/nginx/sites-enabled/
    fi
    
    # Test configuration
    echo "→ Testing nginx configuration..."
    sudo nginx -t
    
    # Reload nginx
    echo "→ Reloading nginx..."
    sudo systemctl reload nginx
    
    echo "✓ Nginx configured successfully!"
}

# Function to setup SSL
setup_ssl() {
    echo ""
    echo "🔒 Setting up SSL..."
    
    if ! command -v certbot &> /dev/null; then
        echo "→ Installing certbot..."
        sudo apt-get update
        sudo apt-get install -y certbot python3-certbot-nginx
    fi
    
    echo "→ Obtaining SSL certificates..."
    echo "⚠️  Run these commands manually:"
    echo "sudo certbot --nginx -d api.quantumfond.com"
    echo "sudo certbot --nginx -d app.quantumfond.com"
}

# Function to verify deployment
verify_deployment() {
    echo ""
    echo "✅ Verifying Deployment..."
    
    # Check backend health
    echo "→ Checking backend health..."
    BACKEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")
    if [ "$BACKEND_HEALTH" = "200" ]; then
        echo "✓ Backend is healthy"
    else
        echo "✗ Backend health check failed (HTTP $BACKEND_HEALTH)"
    fi
    
    # Check frontend
    echo "→ Checking frontend..."
    FRONTEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/health || echo "000")
    if [ "$FRONTEND_HEALTH" = "200" ]; then
        echo "✓ Frontend is healthy"
    else
        echo "✗ Frontend health check failed (HTTP $FRONTEND_HEALTH)"
    fi
    
    # Check systemd service
    echo "→ Checking systemd service..."
    if sudo systemctl is-active --quiet quantumfond-backend; then
        echo "✓ Backend service is running"
    else
        echo "✗ Backend service is not running"
    fi
}

# Main deployment flow
main() {
    # Backup database if production
    if [ "$ENVIRONMENT" = "production" ]; then
        backup_database
    fi
    
    # Deploy components
    deploy_backend
    deploy_frontend
    configure_nginx
    
    # Verify
    verify_deployment
    
    echo ""
    echo "🎉 Deployment Complete!"
    echo "─────────────────────────────────"
    echo "Backend API: https://api.quantumfond.com"
    echo "Frontend App: https://app.quantumfond.com"
    echo ""
    echo "Next steps:"
    echo "1. Configure SSL certificates (see setup_ssl output above)"
    echo "2. Update DNS records to point to this server"
    echo "3. Configure .env file at $BACKEND_DIR/.env"
    echo "4. Test the application: https://app.quantumfond.com"
    echo ""
    echo "Logs:"
    echo "- Backend: journalctl -u quantumfond-backend -f"
    echo "- Nginx: tail -f /var/log/nginx/quantumfond-*.log"
}

# Run main deployment
main
