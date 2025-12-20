#!/bin/bash
# Open port 8080 for Quantum Trader Dashboard

echo "🔓 Opening port 8080 for Dashboard access..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Please run as root (sudo bash open_dashboard_port.sh)"
    exit 1
fi

# Check which firewall is active
if command -v ufw &> /dev/null; then
    echo "📋 Detected UFW firewall"
    
    # Allow port 8080
    ufw allow 8080/tcp comment 'Quantum Trader Dashboard'
    
    # Show status
    echo ""
    echo "✅ UFW rules updated:"
    ufw status | grep 8080
    
elif command -v firewall-cmd &> /dev/null; then
    echo "📋 Detected firewalld"
    
    # Allow port 8080
    firewall-cmd --permanent --add-port=8080/tcp
    firewall-cmd --reload
    
    # Show status
    echo ""
    echo "✅ Firewalld rules updated:"
    firewall-cmd --list-ports | grep 8080
    
else
    # Direct iptables
    echo "📋 Using iptables directly"
    
    # Check if rule already exists
    if iptables -C INPUT -p tcp --dport 8080 -j ACCEPT 2>/dev/null; then
        echo "✅ Port 8080 already open in iptables"
    else
        # Add rule to allow port 8080
        iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
        echo "✅ Added iptables rule for port 8080"
        
        # Save rules (method depends on distro)
        if command -v iptables-save &> /dev/null; then
            if [ -f /etc/iptables/rules.v4 ]; then
                iptables-save > /etc/iptables/rules.v4
                echo "💾 Saved to /etc/iptables/rules.v4"
            elif [ -f /etc/sysconfig/iptables ]; then
                iptables-save > /etc/sysconfig/iptables
                echo "💾 Saved to /etc/sysconfig/iptables"
            else
                echo "⚠️  Rules added but may not persist after reboot"
                echo "💡 Consider saving with: iptables-save > /etc/iptables/rules.v4"
            fi
        fi
    fi
    
    # Show current rules for port 8080
    echo ""
    echo "📊 Current iptables rules for port 8080:"
    iptables -L INPUT -n -v | grep 8080
fi

echo ""
echo "🔍 Verifying port 8080 is listening..."
ss -tuln | grep 8080

echo ""
echo "✅ Port 8080 configuration complete!"
echo "🌐 Dashboard should now be accessible at: http://46.224.116.254:8080"
echo ""
echo "💡 Test with: curl http://localhost:8080/"
