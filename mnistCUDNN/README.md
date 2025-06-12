# MNIST CUDNN 프로젝트

GPU 기반 MNIST 손글씨 숫자 인식을 위한 CUDNN 구현 프로젝트입니다.

## 📁 프로젝트 구조

```
mnistCUDNN/
├── src/                    # 소스 코드 파일들
│   ├── mnistCUDNN.cpp
│   ├── mnistCUDNN_improved.cpp
│   ├── mnistCUDNN_simple.cpp
│   ├── *.h                 # 헤더 파일들
│   └── *.cu                # CUDA 소스 파일들
├── bin/                    # 실행 파일들
│   ├── mnistCUDNN*
│   └── mnist_improved
├── obj/                    # 컴파일된 오브젝트 파일들
│   └── *.o
├── models/                 # 모델 데이터 및 가중치
│   ├── data/              # 기본 모델 가중치
│   ├── data_simple/       # 단순 모델 가중치
│   ├── data_improved/     # 개선된 모델 가중치
│   ├── test_image/        # 테스트 이미지들
│   └── mnist_trainer/     # Python 기반 모델 훈련
├── profiles/              # 성능 프로파일링 데이터
│   ├── *.nsys-rep
│   ├── *.sqlite
│   └── *.csv
├── scripts/               # 빌드 및 실행 스크립트들
│   ├── Makefile*
│   ├── *.sh
│   └── build scripts
├── docs/                  # 문서 파일들
│   ├── *.md
│   └── reports
└── lib/                   # 라이브러리
    └── FreeImage/
```

## 🚀 빌드 방법

```bash
# 개선된 모델 빌드
make

# 단순 모델 빌드  
make -f scripts/Makefile_simple

# 클린
make clean
```

## 🏃 실행 방법

```bash
# 개선된 모델 실행
./bin/mnistCUDNN_improved

# 단순 모델 실행
./bin/mnistCUDNN_simple

# 데모 스크립트 실행
./scripts/demo.sh
```

## 📋 모델 버전

1. **기본 모델** (`mnistCUDNN.cpp`) - 표준 MNIST CNN 구현
2. **단순 모델** (`mnistCUDNN_simple.cpp`) - 경량화된 버전
3. **개선된 모델** (`mnistCUDNN_improved.cpp`) - 배치 정규화 등 고급 기법 적용

## 📊 성능 분석

프로파일링 데이터는 `profiles/` 디렉토리에서 확인할 수 있습니다.
- NVIDIA Nsight Systems 프로파일 (.nsys-rep)
- 성능 데이터베이스 (.sqlite)
- CUDA 트레이스 데이터 (.csv)

## 📚 문서

자세한 문서는 `docs/` 디렉토리를 참조하세요:
- 모델 비교 리포트
- 성능 분석 리포트
- 개선된 모델 가이드

## 🔧 요구사항

- CUDA Toolkit
- CUDNN
- FreeImage 라이브러리
- C++11 지원 컴파일러 