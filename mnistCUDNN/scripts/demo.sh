#!/bin/bash

echo "=== MNIST CUDNN 개선된 모델 데모 ==="
echo ""

# 색상 코드 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_step() {
    echo -e "${BLUE}[단계 $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# CUDA 확인
print_step 1 "CUDA 환경 확인"
if command -v nvcc &> /dev/null; then
    nvcc_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    print_success "CUDA 버전: $nvcc_version"
else
    print_error "CUDA가 설치되지 않았습니다."
    exit 1
fi

# Python 확인
print_step 2 "Python 환경 확인"
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    print_success "$python_version"
else
    print_error "Python 3가 설치되지 않았습니다."
    exit 1
fi

# PyTorch CUDA 확인
echo -n "PyTorch CUDA 지원 확인 중... "
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    print_success "PyTorch에서 CUDA 사용 가능"
else
    print_warning "PyTorch에서 CUDA를 사용할 수 없습니다. CPU로 학습됩니다."
fi

echo ""

# 사용자에게 확인
echo "다음 작업이 수행됩니다:"
echo "1. 개선된 MNIST 모델 학습 (15-20분 소요)"
echo "2. 가중치를 CUDNN 형식으로 내보내기"
echo "3. C++ CUDNN 구현 컴파일"
echo "4. 테스트 실행"
echo ""
read -p "계속하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "데모가 취소되었습니다."
    exit 0
fi

echo ""

# 모델 학습 및 내보내기
print_step 3 "모델 학습 및 가중치 내보내기"
if [ -f "mnist_trainer/train_and_export.sh" ]; then
    ./mnist_trainer/train_and_export.sh
    if [ $? -eq 0 ]; then
        print_success "모델 학습 및 내보내기 완료"
    else
        print_error "모델 학습 실패"
        exit 1
    fi
else
    print_error "학습 스크립트를 찾을 수 없습니다."
    exit 1
fi

echo ""

# 컴파일 - 간단한 개선된 모델
print_step 4 "간단한 개선된 모델 컴파일"
if make -f Makefile_simple_improved clean && make -f Makefile_simple_improved; then
    print_success "간단한 개선된 모델 컴파일 완료"
else
    print_error "간단한 개선된 모델 컴파일 실패"
    exit 1
fi

echo ""

# 완전한 개선된 모델 컴파일 시도
print_step 5 "완전한 개선된 모델 컴파일 시도"
if make -f Makefile_improved_v2 clean && make -f Makefile_improved_v2; then
    print_success "완전한 개선된 모델 컴파일 완료"
    COMPLETE_MODEL_AVAILABLE=true
else
    print_warning "완전한 개선된 모델 컴파일 실패 (CUDNN 호환성 이슈)"
    COMPLETE_MODEL_AVAILABLE=false
fi

echo ""

# 테스트 실행
print_step 6 "개선된 모델 테스트"
if [ -f "./mnistCUDNN_simple_improved" ]; then
    echo "간단한 개선된 모델 테스트 실행 중..."
    ./mnistCUDNN_simple_improved
    
    if [ $? -eq 0 ]; then
        print_success "간단한 개선된 모델 테스트 성공!"
    else
        print_warning "간단한 개선된 모델 테스트에서 일부 문제가 발생했습니다."
    fi
else
    print_error "간단한 개선된 모델 실행 파일을 찾을 수 없습니다."
fi

if [ "$COMPLETE_MODEL_AVAILABLE" = true ] && [ -f "./mnistCUDNN_improved_v2" ]; then
    echo ""
    echo "완전한 개선된 모델 테스트 실행 중..."
    ./mnistCUDNN_improved_v2
    
    if [ $? -eq 0 ]; then
        print_success "완전한 개선된 모델 테스트 성공!"
    else
        print_warning "완전한 개선된 모델 테스트에서 런타임 오류 발생 (CUDNN 호환성 이슈)"
    fi
fi

echo ""

# 배치 테스트 (선택사항)
if [ -d "test_image" ] && [ "$(ls -A test_image/*.pgm 2>/dev/null)" ]; then
    echo "배치 테스트도 실행하시겠습니까? (test_image 디렉토리의 모든 이미지)"
    read -p "(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step 6 "배치 테스트 실행"
        ./mnistCUDN_improved batch
    fi
fi

echo ""
print_success "=== 데모 완료 ==="
echo ""
echo "${GREEN}다음 명령어로 개선된 모델을 사용할 수 있습니다:${NC}"
echo "• ./mnistCUDNN_improved                    # 기본 테스트"
echo "• ./mnistCUDNN_improved image=path.pgm     # 특정 이미지 분류"
echo "• ./mnistCUDNN_improved batch              # 배치 처리"
echo ""
echo "${BLUE}성능 비교를 위해 기존 모델도 실행해보세요:${NC}"
echo "• ./mnistCUDNN                             # 기존 모델"
echo ""
echo "${YELLOW}더 자세한 정보는 README_complete_implementation.md를 참조하세요.${NC}" 