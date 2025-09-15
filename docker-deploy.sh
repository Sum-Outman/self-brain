#!/bin/bash

# Self Brain AGI Docker Deployment Script
# Usage: ./docker-deploy.sh [dev|prod|stop|logs|clean]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default environment
ENV=${1:-dev}

echo -e "${GREEN}Self Brain AGI Docker Deployment${NC}"
echo "Environment: $ENV"

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
}

# Function to deploy development environment
deploy_dev() {
    echo -e "${YELLOW}Deploying development environment...${NC}"
    check_docker
    
    # Create necessary directories
    mkdir -p data logs models uploads static ssl
    
    # Build and start services
    docker-compose -f docker-compose.dev.yml build --no-cache
    docker-compose -f docker-compose.dev.yml up -d
    
    echo -e "${GREEN}Development environment deployed successfully!${NC}"
    echo "Services available at:"
    echo "  - Main Interface: http://localhost:5000"
    echo "  - D Image: http://localhost:5004"
    echo "  - E Video: http://localhost:5005"
    echo "  - F Spatial: http://localhost:5006"
    echo "  - G Sensor: http://localhost:5007"
    echo "  - H Control: http://localhost:5008"
    echo "  - I Knowledge: http://localhost:5009"
    echo "  - J Motion: http://localhost:5010"
    echo "  - K Programming: http://localhost:5011"
}

# Function to deploy production environment
deploy_prod() {
    echo -e "${YELLOW}Deploying production environment...${NC}"
    check_docker
    
    # Create necessary directories
    mkdir -p data logs models uploads static ssl
    
    # Build and start services
    docker-compose -f docker-compose.prod.yml build --no-cache
    docker-compose -f docker-compose.prod.yml up -d
    
    echo -e "${GREEN}Production environment deployed successfully!${NC}"
    echo "Services available at:"
    echo "  - Main Interface: http://localhost:5000"
    echo "  - Nginx Proxy: http://localhost:80"
    echo "  - Monitoring: http://localhost:3000 (Grafana)"
    echo "  - Prometheus: http://localhost:9090"
}

# Function to stop all services
stop_services() {
    echo -e "${YELLOW}Stopping all services...${NC}"
    docker-compose -f docker-compose.dev.yml down || true
    docker-compose -f docker-compose.prod.yml down || true
    docker-compose -f docker-compose.yml down || true
    echo -e "${GREEN}All services stopped.${NC}"
}

# Function to view logs
view_logs() {
    SERVICE=${2:-""}
    if [ -z "$SERVICE" ]; then
        echo -e "${YELLOW}Viewing all logs...${NC}"
        docker-compose -f docker-compose.dev.yml logs -f || docker-compose -f docker-compose.prod.yml logs -f
    else
        echo -e "${YELLOW}Viewing logs for $SERVICE...${NC}"
        docker-compose -f docker-compose.dev.yml logs -f $SERVICE || docker-compose -f docker-compose.prod.yml logs -f $SERVICE
    fi
}

# Function to clean up
clean_up() {
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    stop_services
    docker system prune -f
    docker volume prune -f
    docker image prune -f
    echo -e "${GREEN}Cleanup completed.${NC}"
}

# Function to check service health
check_health() {
    echo -e "${YELLOW}Checking service health...${NC}"
    
    services=("5000" "5004" "5005" "5006" "5007" "5008" "5009" "5010" "5011")
    
    for port in "${services[@]}"; do
        if curl -f http://localhost:$port/health > /dev/null 2>&1; then
            echo -e "${GREEN}Port $port: Healthy${NC}"
        else
            echo -e "${RED}Port $port: Unhealthy${NC}"
        fi
    done
}

# Main script logic
case $ENV in
    "dev")
        deploy_dev
        ;;
    "prod")
        deploy_prod
        ;;
    "stop")
        stop_services
        ;;
    "logs")
        view_logs $2
        ;;
    "clean")
        clean_up
        ;;
    "health")
        check_health
        ;;
    *)
        echo "Usage: $0 [dev|prod|stop|logs|clean|health]"
        echo ""
        echo "Commands:"
        echo "  dev     - Deploy development environment"
        echo "  prod    - Deploy production environment"
        echo "  stop    - Stop all services"
        echo "  logs    - View logs [service_name]"
        echo "  clean   - Clean up Docker resources"
        echo "  health  - Check service health"
        exit 1
        ;;
esac