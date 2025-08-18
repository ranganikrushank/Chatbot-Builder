import hashlib

# Your custom credentials
username = "Krushank_Admin"  # Change this
password = "@Krushank@AtlanciaWorldwide"  # Change this

# Generate hash
password_hash = hashlib.sha256(password.encode()).hexdigest()

print("🔐 Your Admin Credentials:")
print(f"Username: {username}")
print(f"Password: {password}")
print(f"Password Hash: {password_hash}")

print(f"\n.env configuration:")
print(f"ADMIN_USERNAME={username}")
print(f"ADMIN_PASSWORD_HASH={password_hash}")