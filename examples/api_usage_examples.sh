#!/bin/bash

# Resume Parser API Usage Examples - Shell Script
# This script demonstrates how to use the Resume Parser API with curl commands

# Configuration
API_BASE_URL="http://localhost:8001"
API_HOST="localhost"
API_PORT="8001"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if API is running
check_api_status() {
    print_status $BLUE "🔍 Checking API status..."
    
    if curl -s "${API_BASE_URL}/health" > /dev/null; then
        print_status $GREEN "✅ API is running"
        return 0
    else
        print_status $RED "❌ API is not running. Please start the server first:"
        print_status $YELLOW "   python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000"
        return 1
    fi
}

# Function to check vLLM status
check_vllm_status() {
    print_status $BLUE "🔍 Checking vLLM server status..."
    
    response=$(curl -s "${API_BASE_URL}/vllm/status")
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ vLLM Status:"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        print_status $RED "❌ Failed to check vLLM status"
    fi
}

# Example 1: Basic resume parsing
example_1_basic_parsing() {
    print_status $BLUE "📋 Example 1: Basic Resume Parsing"
    echo "=================================================="
    
    # Check if sample file exists
    if [ ! -f "examples/sample_resume.pdf" ]; then
        print_status $YELLOW "⚠️  Sample file not found. Creating a test file..."
        mkdir -p examples
        cat > examples/sample_resume.txt << 'EOF'
JOHN DOE
Software Engineer
john.doe@email.com | (555) 123-4567

SUMMARY
Experienced software engineer with 5+ years in full-stack development.

EXPERIENCE
Senior Developer | Tech Corp | 2020-2023
- Led development of web applications
- Mentored junior developers

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2018

SKILLS
Python, JavaScript, React, Node.js, Docker, AWS
EOF
        print_status $GREEN "✅ Created sample file: examples/sample_resume.txt"
    fi
    
    # Parse the resume
    print_status $BLUE "🔄 Parsing resume..."
    
    if [ -f "examples/sample_resume.pdf" ]; then
        response=$(curl -s -X POST "${API_BASE_URL}/parse" \
            -F "file=@examples/sample_resume.pdf" \
            -F "use_llm=true")
    else
        response=$(curl -s -X POST "${API_BASE_URL}/parse" \
            -F "file=@examples/sample_resume.txt" \
            -F "use_llm=true")
    fi
    
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ Parse successful:"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        print_status $RED "❌ Parse failed"
    fi
}

# Example 2: Parse without LLM enhancement
example_2_without_llm() {
    print_status $BLUE "📋 Example 2: Parsing Without LLM Enhancement"
    echo "=================================================="
    
    print_status $BLUE "🔄 Parsing resume without LLM..."
    
    if [ -f "examples/sample_resume.pdf" ]; then
        response=$(curl -s -X POST "${API_BASE_URL}/parse" \
            -F "file=@examples/sample_resume.pdf" \
            -F "use_llm=false")
    else
        response=$(curl -s -X POST "${API_BASE_URL}/parse" \
            -F "file=@examples/sample_resume.txt" \
            -F "use_llm=false")
    fi
    
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ Parse successful (without LLM):"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        print_status $RED "❌ Parse failed"
    fi
}

# Example 3: Extract specific information
example_3_specific_extraction() {
    print_status $BLUE "📋 Example 3: Specific Information Extraction"
    echo "=================================================="
    
    # Extract different types of information
    info_types=("skills" "experience" "education" "contact")
    
    for info_type in "${info_types[@]}"; do
        print_status $BLUE "🔍 Extracting $info_type..."
        
        if [ -f "examples/sample_resume.pdf" ]; then
            response=$(curl -s -X POST "${API_BASE_URL}/extract/specific" \
                -F "file=@examples/sample_resume.pdf" \
                -F "info_type=$info_type")
        else
            response=$(curl -s -X POST "${API_BASE_URL}/extract/specific" \
                -F "file=@examples/sample_resume.txt" \
                -F "info_type=$info_type")
        fi
        
        if [ $? -eq 0 ]; then
            print_status $GREEN "✅ $info_type extracted:"
            echo "$response" | jq '.' 2>/dev/null || echo "$response"
        else
            print_status $RED "❌ Failed to extract $info_type"
        fi
        echo
    done
}

# Example 4: Batch processing
example_4_batch_processing() {
    print_status $BLUE "📋 Example 4: Batch Resume Processing"
    echo "=================================================="
    
    # Create multiple sample files for batch processing
    mkdir -p examples
    
    # Create sample resume 1
    cat > examples/resume1.txt << 'EOF'
JANE SMITH
Data Scientist
jane.smith@email.com

EXPERIENCE
Data Scientist | AI Corp | 2021-2023
- Built ML models for prediction
- Analyzed large datasets

SKILLS
Python, R, TensorFlow, SQL, Pandas
EOF

    # Create sample resume 2
    cat > examples/resume2.txt << 'EOF'
MIKE JOHNSON
DevOps Engineer
mike.johnson@email.com

EXPERIENCE
DevOps Engineer | Cloud Inc | 2019-2023
- Managed Kubernetes clusters
- Automated deployment pipelines

SKILLS
Docker, Kubernetes, AWS, Terraform, Jenkins
EOF

    print_status $BLUE "🔄 Processing batch of resumes..."
    
    # Use curl to send multiple files
    response=$(curl -s -X POST "${API_BASE_URL}/parse/batch" \
        -F "files=@examples/resume1.txt" \
        -F "files=@examples/resume2.txt" \
        -F "use_llm=true")
    
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ Batch processing successful:"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        print_status $RED "❌ Batch processing failed"
    fi
}

# Example 5: Error handling
example_5_error_handling() {
    print_status $BLUE "📋 Example 5: Error Handling Examples"
    echo "=================================================="
    
    # Test with non-existent file
    print_status $BLUE "🔍 Testing with non-existent file..."
    response=$(curl -s -X POST "${API_BASE_URL}/parse" \
        -F "file=@non_existent_file.pdf" \
        -F "use_llm=true")
    
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ Error handling test:"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        print_status $RED "❌ Error handling test failed"
    fi
    
    # Test with unsupported file type
    print_status $BLUE "🔍 Testing with unsupported file type..."
    echo "test content" > examples/test.xyz
    
    response=$(curl -s -X POST "${API_BASE_URL}/parse" \
        -F "file=@examples/test.xyz" \
        -F "use_llm=true")
    
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ Unsupported file type test:"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        print_status $RED "❌ Unsupported file type test failed"
    fi
}

# Example 6: Health check and monitoring
example_6_health_monitoring() {
    print_status $BLUE "📋 Example 6: Health Check and Monitoring"
    echo "=================================================="
    
    # Check API health
    print_status $BLUE "🔍 Checking API health..."
    health_response=$(curl -s "${API_BASE_URL}/health")
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ API Health:"
        echo "$health_response" | jq '.' 2>/dev/null || echo "$health_response"
    else
        print_status $RED "❌ Health check failed"
    fi
    
    # Check vLLM status
    check_vllm_status
}

# Example 7: Performance testing
example_7_performance_test() {
    print_status $BLUE "📋 Example 7: Performance Testing"
    echo "=================================================="
    
    if [ ! -f "examples/sample_resume.txt" ]; then
        print_status $YELLOW "⚠️  Sample file not found. Creating one..."
        example_1_basic_parsing > /dev/null 2>&1
    fi
    
    print_status $BLUE "⏱️  Testing API response time..."
    
    # Test response time
    start_time=$(date +%s.%N)
    response=$(curl -s -X POST "${API_BASE_URL}/parse" \
        -F "file=@examples/sample_resume.txt" \
        -F "use_llm=true")
    end_time=$(date +%s.%N)
    
    execution_time=$(echo "$end_time - $start_time" | bc)
    
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ Performance test completed:"
        print_status $GREEN "   Response time: ${execution_time} seconds"
        print_status $GREEN "   Status: Success"
    else
        print_status $RED "❌ Performance test failed"
    fi
}

# Example 8: Advanced curl options
example_8_advanced_curl() {
    print_status $BLUE "📋 Example 8: Advanced cURL Options"
    echo "=================================================="
    
    print_status $BLUE "🔧 Testing with verbose output and custom headers..."
    
    # Test with verbose output
    response=$(curl -v -X POST "${API_BASE_URL}/parse" \
        -F "file=@examples/sample_resume.txt" \
        -F "use_llm=true" \
        -H "Accept: application/json" \
        -H "User-Agent: ResumeParser-Test/1.0" \
        2>&1)
    
    print_status $GREEN "✅ Advanced cURL test completed"
    print_status $BLUE "   Verbose output shows request/response details"
}

# Main function
main() {
    print_status $GREEN "🚀 Resume Parser API Usage Examples - Shell Script"
    echo "=================================================="
    
    # Check if API is running
    if ! check_api_status; then
        exit 1
    fi
    
    # Check if jq is available for pretty printing
    if ! command -v jq &> /dev/null; then
        print_status $YELLOW "⚠️  jq not found. Install jq for better JSON formatting:"
        print_status $YELLOW "   sudo apt-get install jq"
    fi
    
    # Run examples
    example_1_basic_parsing
    echo
    example_2_without_llm
    echo
    example_3_specific_extraction
    echo
    example_4_batch_processing
    echo
    example_5_error_handling
    echo
    example_6_health_monitoring
    echo
    example_7_performance_test
    echo
    example_8_advanced_curl
    
    print_status $GREEN "✅ All examples completed!"
    print_status $BLUE "📝 For more information, see the Python examples:"
    print_status $BLUE "   python examples/api_usage_examples.py"
}

# Run main function
main "$@" 