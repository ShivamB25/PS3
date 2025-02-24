# Quick Start Guide

This guide provides step-by-step instructions for setting up and testing the P2P-enhanced price optimization system.

## Prerequisites

- Python 3.13 or higher
- Node.js and npm
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install Backend Dependencies:
   ```bash
   cd Backend
   python -m pip install -e .
   ```

3. Install Frontend Dependencies:
   ```bash
   cd Frontend
   npm install
   ```

## Starting the Servers

### Backend Server

1. Navigate to the Backend directory:
   ```bash
   cd Backend
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
   ```

   The server will be available at:
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Frontend Development Server

1. Navigate to the Frontend directory:
   ```bash
   cd Frontend
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

   The application will be available at http://localhost:3000

## Testing the P2P Features

1. Access the Application:
   - Open http://localhost:3000 in your browser
   - Select a model from the list
   - Scroll down to the "P2P Network" section

2. Test Network Status:
   - Check connection status
   - View network mode
   - Monitor peer count
   - Verify last sync time
   - Check network health

3. Configure Network Settings:
   - Try different network modes:
     * Private (Single Organization)
     * Consortium (Trusted Group)
     * Public (Open Market)
   - Adjust privacy settings:
     * Data anonymization
     * Connection encryption
     * Data sharing preferences

4. Monitor Market Insights:
   - View average prices
   - Track price trends
   - Check network activity
   - Review market signals

5. Test Collaborative Features:
   - Generate predictions
   - Observe network-enhanced results
   - Monitor exploration bonuses
   - Check confidence levels

## Troubleshooting

### Common Issues

1. Backend Server Won't Start:
   - Check Python version (`python --version`)
   - Verify all dependencies are installed
   - Check port 8000 is available
   - Look for error messages in terminal

2. Frontend Server Issues:
   - Verify Node.js installation (`node --version`)
   - Check npm dependencies (`npm install`)
   - Clear npm cache if needed (`npm cache clean --force`)
   - Look for errors in browser console

3. Network Connection Issues:
   - Check backend server is running
   - Verify network configuration
   - Check browser console for errors
   - Ensure correct ports are open

### Error Messages

1. Backend Errors:
   ```
   Error: Address already in use
   Solution: Change port or stop process using port 8000
   ```

2. Frontend Errors:
   ```
   Error: Failed to fetch network status
   Solution: Ensure backend server is running
   ```

## Verification Steps

1. Network Status:
   - Connection indicator should be green
   - Peer count should update
   - Health status should be visible
   - Last sync time should update

2. Market Insights:
   - Price trends should display
   - Network activity should update
   - Market signals should appear
   - Health score should be visible

3. Configuration Changes:
   - Settings should apply immediately
   - Network should reconnect as needed
   - Privacy rules should be enforced
   - Data sharing should respect settings

## Next Steps

1. Read detailed documentation:
   - [Implementation Plan](implementation_plan.md)
   - [Technical Architecture](technical_architecture.md)
   - [User Guide](user_guide.md)
   - [Testing Guide](testing_guide.md)

2. Explore advanced features:
   - Custom network configurations
   - Advanced privacy settings
   - Market analysis tools
   - Performance monitoring

3. Monitor system performance:
   - Check network health
   - Monitor data synchronization
   - Track prediction accuracy
   - Analyze market insights

For more detailed information, refer to the full documentation in the `docs/pear` directory.
