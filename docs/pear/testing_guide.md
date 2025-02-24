# Testing Guide for P2P Functionality

This guide explains how to test the P2P (Peer-to-Peer) features of the price optimization system.

## Prerequisites

1. Backend server running:
   ```bash
   cd Backend
   python -m pip install -e .
   python src/server.py
   ```

2. Frontend development server:
   ```bash
   cd Frontend
   npm install
   npm run dev
   ```

## Testing Network Status

1. Open http://localhost:3000 in your browser
2. Select a model from the list
3. Scroll down to the "P2P Network" section
4. The NetworkStatus component should show:
   - Connection status
   - Network mode
   - Number of connected peers
   - Last sync time
   - Network health

## Testing Network Configuration

1. In the NetworkConfig component:
   - Try changing the network mode (Private/Consortium/Public)
   - Toggle privacy settings
   - Modify data sharing preferences
2. Changes should be reflected in the NetworkStatus component
3. Verify settings persist after page refresh

## Testing Market Insights

1. The NetworkInsights component should display:
   - Average price across the network
   - Price trends
   - Network activity metrics
   - Recent market signals
2. Data should update automatically every 30 seconds

## Testing P2P Price Memory

1. Generate predictions for a product
2. Observe how the model incorporates network insights:
   - Exploration bonuses should be influenced by peer data
   - Price recommendations should consider market trends
   - Confidence levels should reflect network consensus

## Testing Different Network Modes

### Private Mode
1. Set network mode to "Private"
2. Verify that:
   - Only internal nodes are visible
   - Full data sharing is enabled
   - High security indicators are shown

### Consortium Mode
1. Set network mode to "Consortium"
2. Verify that:
   - Trusted peers are visible
   - Selective data sharing is enforced
   - Privacy protections are active

### Public Mode
1. Set network mode to "Public"
2. Verify that:
   - All peers are visible
   - Basic market signals are shared
   - Maximum privacy measures are in place

## Troubleshooting

### Network Connection Issues
1. Check Backend server logs
2. Verify network configuration
3. Check browser console for errors
4. Ensure correct ports are open (8000 for Backend, 3000 for Frontend)

### Data Synchronization Issues
1. Check last sync time in NetworkStatus
2. Verify network health indicator
3. Check Backend logs for sync errors
4. Try manual sync through NetworkConfig

### Privacy Settings Issues
1. Verify privacy configuration in Backend
2. Check data sharing rules enforcement
3. Monitor network traffic for data leaks
4. Test data anonymization

## Performance Testing

### Network Latency
1. Monitor sync intervals
2. Check response times for:
   - Status updates
   - Configuration changes
   - Market insights

### Data Processing
1. Test with increasing peer counts
2. Monitor memory usage
3. Check CPU utilization
4. Verify data aggregation speed

## Security Testing

### Authentication
1. Verify peer authentication
2. Test connection encryption
3. Check certificate validation

### Data Protection
1. Test data anonymization
2. Verify encryption at rest
3. Check secure transmission
4. Test access controls

## Expected Results

### Network Status
- Connection indicator should be green when connected
- Peer count should be accurate
- Health status should reflect network conditions
- Last sync time should update regularly

### Market Insights
- Price trends should be clearly visible
- Network activity should be current
- Market signals should be relevant
- Health score should be accurate

### Configuration Changes
- Settings should apply immediately
- Network should reconnect as needed
- Privacy rules should be enforced
- Data sharing should respect settings

## Common Issues

1. Connection Failures
   - Check network connectivity
   - Verify server status
   - Check port availability
   - Review firewall settings

2. Data Sync Issues
   - Check network stability
   - Verify peer availability
   - Review sync logs
   - Check data integrity

3. Privacy Violations
   - Review privacy settings
   - Check data anonymization
   - Verify encryption
   - Monitor data sharing

## Reporting Issues

When reporting issues, include:
1. Network mode
2. Privacy settings
3. Error messages
4. Relevant logs
5. Steps to reproduce
6. Expected vs actual behavior
