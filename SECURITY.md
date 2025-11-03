# Security Policy

## Supported Versions

Currently, this project is in active development. Security updates will be provided for the latest version.

## Reporting a Vulnerability

If you discover a security vulnerability, please email [your-email] with details. Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

We will respond within 48 hours and work with you to address the issue.

## Security Considerations

### API Keys
- Never commit API keys to the repository
- Use environment variables or secure config files (not tracked by Git)
- Rotate keys regularly

### Data Privacy
- This project uses public market data (Binance API)
- No personal or sensitive data is processed

### Code Security
- All dependencies from trusted sources (PyPI)
- Regular dependency updates recommended
- Code review process for contributions

## Best Practices

1. **Never commit:**
   - API keys or secrets
   - Private trading strategies
   - Personal data

2. **Use environment variables:**
   ```bash
   export BINANCE_API_KEY="your-key"
   export BINANCE_SECRET_KEY="your-secret"
   ```

3. **Review dependencies:**
   - Regularly update `requirements.txt`
   - Audit for known vulnerabilities

