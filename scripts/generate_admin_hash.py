#!/usr/bin/env python3
"""
Generate secure admin password hash for IngestForge API.

This utility helps administrators create a bcrypt password hash
for use with the INGESTFORGE_ADMIN_PASSWORD_HASH environment variable.

Usage:
    python scripts/generate_admin_hash.py
    
Security:
    - Uses bcrypt for password hashing (industry standard)
    - Automatically generates secure JWT secret key
    - Password input is masked (not echoed to terminal)
    - Follows NASA JPL Rule #7 (parameter validation)
"""

import sys
import secrets
from getpass import getpass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def generate_jwt_secret() -> str:
    """Generate a cryptographically secure JWT secret key."""
    return secrets.token_urlsafe(32)


def generate_password_hash(password: str) -> str:
    """Generate bcrypt hash from plain password."""
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    return pwd_context.hash(password)


def main():
    """Interactive password hash generator."""
    print("=" * 70)
    print("IngestForge Admin Credential Generator")
    print("=" * 70)
    print()
    print("This tool generates secure credentials for the IngestForge API.")
    print()

    # Get username
    print("Step 1: Admin Username")
    username = input("Enter admin username (default: admin): ").strip()
    if not username:
        username = "admin"
    print(f"✓ Username: {username}")
    print()

    # Get password
    print("Step 2: Admin Password")
    print("Password requirements:")
    print("  - Minimum 8 characters")
    print("  - Use a strong, unique password")
    print()

    while True:
        password = getpass("Enter admin password: ")
        if not password:
            print("Error: Password cannot be empty")
            continue
        if len(password) < 8:
            print("Error: Password must be at least 8 characters")
            continue

        password_confirm = getpass("Confirm password: ")
        if password != password_confirm:
            print("Error: Passwords do not match. Please try again.")
            continue

        break

    # Generate hash
    print()
    print("Generating secure password hash...")
    password_hash = generate_password_hash(password)
    print("✓ Password hash generated")
    print()

    # Generate JWT secret
    print("Generating JWT secret key...")
    jwt_secret = generate_jwt_secret()
    print("✓ JWT secret key generated")
    print()

    # Display results
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print()
    print("Add these environment variables to your .env file:")
    print()
    print(f"INGESTFORGE_ADMIN_USERNAME={username}")
    print(f"INGESTFORGE_ADMIN_PASSWORD_HASH={password_hash}")
    print(f"INGESTFORGE_JWT_SECRET_KEY={jwt_secret}")
    print()
    print("=" * 70)
    print()
    print("⚠️  SECURITY WARNING:")
    print("  - Store these values securely in your .env file")
    print("  - Never commit .env to version control")
    print("  - Keep the password hash secret and secure")
    print("  - Rotate credentials periodically")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
