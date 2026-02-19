# IngestForge Security Setup Guide

## Overview

IngestForge API uses secure, environment-based authentication to protect your data. This guide explains how to configure authentication credentials properly.

## Critical Security Notice

**NEVER use hardcoded credentials in production.** All authentication credentials must be configured via environment variables.

## Quick Start

### 1. Generate Secure Credentials

Use the provided utility script to generate secure credentials:

```bash
python scripts/generate_admin_hash.py
```

This will interactively prompt you for:
- Admin username (default: `admin`)
- Admin password (minimum 8 characters)

The script outputs three environment variables that you'll need.

### 2. Configure Environment Variables

Copy the generated variables to your `.env` file:

```bash
# Authentication & Security
INGESTFORGE_ADMIN_USERNAME=admin
INGESTFORGE_ADMIN_PASSWORD_HASH=$2b$12$...  # Generated bcrypt hash
INGESTFORGE_JWT_SECRET_KEY=...  # Generated secret key
```

### 3. Verify Configuration

Start the API server:

```bash
python -m ingestforge.cli.commands.api
```

Test authentication:

```bash
curl -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=your_password"
```

You should receive a JWT token in response.

## Manual Configuration

### Generate JWT Secret Key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Generate Password Hash

```bash
python -c "from passlib.context import CryptContext; pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto'); import getpass; print(pwd_context.hash(getpass.getpass('Password: ')))"
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `INGESTFORGE_ADMIN_USERNAME` | Yes | Admin username for API access |
| `INGESTFORGE_ADMIN_PASSWORD_HASH` | Yes | Bcrypt hash of admin password |
| `INGESTFORGE_JWT_SECRET_KEY` | Yes (Production) | Secret key for JWT token signing |

## Security Best Practices

### Password Requirements

- **Minimum Length**: 8 characters (16+ recommended)
- **Complexity**: Use a mix of uppercase, lowercase, numbers, and symbols
- **Uniqueness**: Never reuse passwords from other systems
- **Storage**: Use a password manager to generate and store passwords

### Secret Key Management

- **Generation**: Always use cryptographically secure random generation
- **Rotation**: Rotate JWT secret keys periodically (invalidates all tokens)
- **Storage**: Store in `.env` file, never commit to version control

### Deployment Security

#### Development
```bash
# Development .env (local only)
INGESTFORGE_ADMIN_USERNAME=admin
INGESTFORGE_ADMIN_PASSWORD_HASH=$2b$12$...
INGESTFORGE_JWT_SECRET_KEY=dev_secret_key_change_in_production
```

#### Production
```bash
# Production .env (secure server)
INGESTFORGE_ADMIN_USERNAME=admin_prod
INGESTFORGE_ADMIN_PASSWORD_HASH=$2b$12$...
INGESTFORGE_JWT_SECRET_KEY=<cryptographically-secure-random-key>
```

**Production checklist:**
- [ ] Strong, unique admin password (16+ characters)
- [ ] Cryptographically secure JWT secret key
- [ ] `.env` file has restricted permissions (chmod 600)
- [ ] Credentials stored in secure secrets manager
- [ ] API served over HTTPS only
- [ ] CORS properly configured for production domains

### File Permissions

Restrict access to configuration files:

```bash
# Linux/macOS
chmod 600 .env

# Verify
ls -la .env
# Should show: -rw------- (owner read/write only)
```

## Troubleshooting

### "Authentication not configured" Error

**Cause**: Missing environment variables

**Solution**: Ensure `INGESTFORGE_ADMIN_USERNAME` and `INGESTFORGE_ADMIN_PASSWORD_HASH` are set in `.env`

### "Invalid credentials" Error

**Cause**: Username or password mismatch

**Solution**: 
1. Verify username matches `INGESTFORGE_ADMIN_USERNAME`
2. Verify password hash was generated correctly
3. Check for typos in `.env` file

### Token Verification Fails

**Cause**: JWT secret key mismatch or expired token

**Solution**:
1. Verify `INGESTFORGE_JWT_SECRET_KEY` is consistent
2. Tokens expire after 24 hours - request new token
3. If secret key changed, all existing tokens are invalid

## Migration from Hardcoded Credentials

If upgrading from a version with hardcoded credentials:

1. **Generate new credentials** using `scripts/generate_admin_hash.py`
2. **Add to .env file** - never use old hardcoded password
3. **Restart API server** to load new environment variables
4. **Update client applications** with new credentials
5. **Verify** old credentials no longer work

## Password Rotation

To change the admin password:

1. Generate new password hash:
```bash
python scripts/generate_admin_hash.py
```

2. Update `.env` with new `INGESTFORGE_ADMIN_PASSWORD_HASH`

3. Restart API server:
```bash
# Stop existing server (Ctrl+C)
# Start new server
python -m ingestforge.cli.commands.api
```

4. Update all client applications with new password

## Security Audit

Verify your configuration is secure:

```bash
# Check for hardcoded credentials in codebase
rg -i "password.*=.*['\"].*['\"]" --type py

# Verify .env is gitignored
git check-ignore .env

# Check file permissions
ls -la .env
```

## Support

For security issues, contact the development team privately. Do not report security vulnerabilities in public issues.

## References

- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [NIST Password Guidelines](https://pages.nist.gov/800-63-3/sp800-63b.html)
- [Bcrypt Documentation](https://passlib.readthedocs.io/en/stable/lib/passlib.hash.bcrypt.html)
