# Tech Stack Documentation

## Frontend Stack

### Core Technologies
- **Next.js 14** - React framework with server-side rendering and API routes
- **TypeScript** - For type-safe development
- **React 18** - UI library with hooks and concurrent features

### UI Components & Styling
- **Tailwind CSS** - Utility-first CSS framework
- **Radix UI** - Headless UI components library with extensive component set:
  - Form controls (inputs, buttons, selects)
  - Navigation components
  - Overlay components (modals, popovers)
  - Data display components
- **Recharts** - For data visualization and charts

### State Management & Forms
- **React Hook Form** - Form state management and validation
- **Zod** - TypeScript-first schema validation

## Backend Stack

### Core Technologies
- **FastAPI** - Modern Python web framework for building APIs
- **Uvicorn** - ASGI server implementation
- **Python 3.13+** - Latest Python version with modern features

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing and array operations

## Machine Learning Stack

### Core ML Framework
- **PyTorch** - Deep learning framework
  - Custom Actor-Critic networks
  - SAC (Soft Actor-Critic) implementation
  - Layer normalization and dropout for stability

### ML Components
- **PyTorch Forecasting** - Time series forecasting tools
- **Scikit-learn** - For data preprocessing and metrics
- **Gymnasium** - For reinforcement learning environments

### ML Features
- Price optimization using reinforcement learning
- Time series forecasting
- Custom price memory implementation
- Exploration vs exploitation strategies

## Development & Build Tools

### Frontend
- **Bun** - JavaScript runtime and package manager
- **ESLint** - JavaScript/TypeScript linting
- **Prettier** - Code formatting
- **PostCSS** - CSS processing

### Backend
- **Black** - Python code formatting
- **Weights & Biases** - ML experiment tracking

## Key Features & Architecture

### Frontend Architecture
- Next.js App Router for routing and API endpoints
- Component-driven development with reusable UI components
- Type-safe API integration
- Responsive design with mobile-first approach

### Backend Architecture
- RESTful API design
- Model registry for managing trained models
- Data preprocessing pipeline
- Async request handling

### ML Architecture
- Actor-Critic architecture with temporal attention
- Soft Actor-Critic (SAC) implementation for price optimization
- Custom environment for price-sales optimization
- Model checkpointing and versioning
- Exploration strategies with temporal context

### Data Flow
- CSV data ingestion
- Real-time price predictions
- Historical data visualization
- Model performance tracking

### Integration Points
- Frontend-Backend: REST API endpoints
- Data Processing: CSV file handling
- Model Training: Async training pipeline
- Visualization: Real-time data updates

## Infrastructure

### Development
- Git version control
- Local development server
- Hot reloading for frontend
- Async API endpoints

### Data Storage
- File-based model registry
- CSV data storage
- Model checkpoints
- Training metadata storage

This tech stack enables a modern, scalable application with robust machine learning capabilities, focusing on price optimization and data visualization while maintaining code quality and developer experience.