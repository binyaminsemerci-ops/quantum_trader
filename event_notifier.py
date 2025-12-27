#!/usr/bin/env python3
"""
🔔 EVENT NOTIFIER
Monitors for specific events and sends notifications
"""
import subprocess
import json
import time
from datetime import datetime, timezone, timedelta
from playsound import playsound
import os

class EventMonitor:
    def __init__(self):
        self.last_training_reward = None
        self.last_pil_count = 0
        self.last_close_count = 0
        self.notifications = []
    
    def get_last_training(self):
        """Get last training run"""
        try:
            result = subprocess.run(
                ["docker", "logs", "quantum_backend", "--tail", "1000"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            training_lines = [line for line in result.stdout.split('\n') 
                             if 'Training completed' in line and 'RLv3' in line]
            
            if training_lines:
                last_line = training_lines[-1]
                data = json.loads(last_line)
                return {
                    'timestamp': datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                    'final_reward': data.get('final_reward', 0),
                }
        except:
            pass
        return None
    
    def get_pil_count(self):
        """Get PIL classification count"""
        try:
            result = subprocess.run(
                ["docker", "exec", "quantum_backend", "python", "-c",
                 "from backend.services.position_intelligence import get_position_intelligence; "
                 "pil = get_position_intelligence(); "
                 "print(len(pil.classifications))"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        return 0
    
    def get_close_count(self):
        """Get position close count from logs"""
        try:
            result = subprocess.run(
                ["docker", "logs", "quantum_backend", "--tail", "500"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            closes = 0
            for line in result.stdout.split('\n'):
                if 'TRADE_EXIT_LOGGED' in line or 'position closed' in line.lower():
                    closes += 1
            
            return closes
        except:
            return 0
    
    def notify(self, title, message, sound=False):
        """Send notification"""
        timestamp = datetime.now(timezone.utc).strftime('%H:%M:%S')
        notification = f"[{timestamp}] {title}: {message}"
        self.notifications.append(notification)
        
        print("\n" + "=" * 80)
        print(f"🔔 NOTIFICATION: {title}")
        print("-" * 80)
        print(message)
        print("=" * 80 + "\n")
        
        # Write to file
        with open('learning_events.log', 'a') as f:
            f.write(f"{notification}\n")
        
        # Play sound if available
        if sound:
            try:
                # You can add a .wav file for notifications
                # playsound('notification.wav')
                pass
            except:
                pass
    
    def check_training(self):
        """Check for new training completion"""
        training = self.get_last_training()
        
        if training:
            if self.last_training_reward is None:
                self.last_training_reward = training['final_reward']
                return False
            
            if training['final_reward'] != self.last_training_reward:
                # New training completed!
                reward_change = training['final_reward'] - self.last_training_reward
                self.notify(
                    "RL v3 TRAINING COMPLETED",
                    f"New training run finished!\n"
                    f"  Previous reward: {self.last_training_reward:,.0f}\n"
                    f"  Current reward:  {training['final_reward']:,.0f}\n"
                    f"  Change: {reward_change:+,.0f} ({(reward_change/self.last_training_reward)*100:+.1f}%)",
                    sound=True
                )
                self.last_training_reward = training['final_reward']
                return True
        
        return False
    
    def check_pil(self):
        """Check for new PIL classifications"""
        pil_count = self.get_pil_count()
        
        if pil_count > self.last_pil_count:
            new_classifications = pil_count - self.last_pil_count
            self.notify(
                "PIL CLASSIFICATION",
                f"Position(s) classified!\n"
                f"  Previous: {self.last_pil_count}\n"
                f"  Current:  {pil_count}\n"
                f"  New: +{new_classifications}",
                sound=True
            )
            self.last_pil_count = pil_count
            return True
        
        return False
    
    def check_closes(self):
        """Check for position closes"""
        close_count = self.get_close_count()
        
        if close_count > self.last_close_count:
            new_closes = close_count - self.last_close_count
            self.notify(
                "POSITION CLOSED",
                f"Position(s) closed - Learning events triggered!\n"
                f"  Previous: {self.last_close_count}\n"
                f"  Current:  {close_count}\n"
                f"  New closes: +{new_closes}\n\n"
                f"Learning systems activated:\n"
                f"  ✅ PIL classification\n"
                f"  ✅ PAL analysis\n"
                f"  ✅ Meta-Strategy update\n"
                f"  ✅ TP Tracker recording",
                sound=True
            )
            self.last_close_count = close_count
            return True
        
        return False
    
    def monitor(self):
        """Main monitoring loop"""
        print("=" * 80)
        print("🔔 EVENT NOTIFIER - MONITORING FOR LEARNING EVENTS")
        print("=" * 80)
        print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()
        print("Monitoring for:")
        print("  1. RL v3 training completion")
        print("  2. PIL classifications (first trade classified)")
        print("  3. Position closes (learning triggers)")
        print()
        print("Notifications will be logged to: learning_events.log")
        print("Press Ctrl+C to stop")
        print("=" * 80)
        print()
        
        # Initialize
        self.last_training_reward = self.get_last_training()['final_reward'] if self.get_last_training() else None
        self.last_pil_count = self.get_pil_count()
        self.last_close_count = self.get_close_count()
        
        print(f"Initial state:")
        print(f"  Last training reward: {self.last_training_reward}")
        print(f"  PIL classifications: {self.last_pil_count}")
        print(f"  Position closes: {self.last_close_count}")
        print()
        print("Monitoring started...")
        print()
        
        check_count = 0
        
        try:
            while True:
                check_count += 1
                
                # Check all events
                training_event = self.check_training()
                pil_event = self.check_pil()
                close_event = self.check_closes()
                
                # Status update every 30 checks (5 minutes at 10s interval)
                if check_count % 30 == 0:
                    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                          f"Status: Monitoring... (checks: {check_count}, "
                          f"events: {len(self.notifications)})")
                
                # Wait before next check
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("MONITORING STOPPED")
            print("=" * 80)
            print(f"Total checks: {check_count}")
            print(f"Total events detected: {len(self.notifications)}")
            print()
            
            if self.notifications:
                print("Event Summary:")
                for notification in self.notifications:
                    print(f"  • {notification}")
            
            print()
            print(f"Full log saved to: learning_events.log")
            print("=" * 80)

if __name__ == "__main__":
    monitor = EventMonitor()
    monitor.monitor()
