# Improved MNIST CUDNN Implementation

이 프로젝트는 기존의 단순한 MNIST CUDNN 구현을 개선하여 더 깊고 복잡한 CNN 아키텍처를 사용합니다.

## 🚀 주요 개선사항

### 기존 모델 vs 개선된 모델

| 구성요소 | 기존 모델 | 개선된 모델 |
|---------|----------|------------|
| Conv 레이어 | 2개 (20, 50 채널) | 6개 (32, 32, 64, 64, 128, 128 채널) |
| Kernel 크기 | 5x5 | 3x3 (더 효율적) |
| Batch Normalization | ❌ | ✅ |
| FC 레이어 | 2개 (500, 10) | 3개 (512, 256, 10) |
| Dropout | ❌ | ✅ (훈련 시) |
| Data Augmentation | ❌ | ✅ |
| 예상 정확도 | ~98% | ~99.5%+ |

## 📋 시스템 요구사항

### 필수 요구사항
- **NVIDIA GPU** (CUDA Compute Capability 3.0+)
- **CUDA Toolkit** (8.0+)
- **cuDNN** (v5.0+)
- **Python 3.6+**
- **GCC/G++** (C++11 지원)

### Python 패키지
- PyTorch 1.12.0+
- torchvision 0.13.0+
- numpy 1.21.0+
- matplotlib 3.5.0+

## 🛠️ 설치 및 사용법

### 1단계: 환경 설정
```bash
# CUDA와 cuDNN이 설치되어 있는지 확인
nvcc --version
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 2단계: 새로운 모델 학습
```bash
# 스크립트 실행 권한 부여
chmod +x mnist_trainer/train_and_export.sh

# 모델 학습 및 가중치 내보내기 (15-20분 소요)
./mnist_trainer/train_and_export.sh
```

이 스크립트는 다음 작업을 수행합니다:
- 필요한 Python 패키지 설치
- 개선된 CNN 모델 학습 (20 에포크)
- 가중치를 CUDNN 호환 바이너리 파일로 내보내기
- `data_improved/` 디렉토리에 모든 가중치 파일 저장

### 3단계: CUDNN 구현 컴파일
```bash
# 개선된 모델 컴파일
make -f Makefile_improved

# 또는 디버그 모드로 컴파일
make -f Makefile_improved DEBUG=1
```

### 4단계: 실행
```bash
# 개선된 모델로 추론 실행
./mnistCUDNN_improved

# 특정 이미지 분류
./mnistCUDNN_improved image=test_image/digit_7.pgm

# 배치 처리 (test_image 디렉토리의 모든 .pgm 파일)
./mnistCUDNN_improved batch
```

## 🏗️ 아키텍처 상세

### 개선된 CNN 구조
```
입력: 28x28x1 (MNIST 이미지)
├── Conv1 (1→32, 3x3) + BatchNorm + ReLU
├── Conv2 (32→32, 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)
├── Conv3 (32→64, 3x3) + BatchNorm + ReLU  
├── Conv4 (64→64, 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)
├── Conv5 (64→128, 3x3) + BatchNorm + ReLU
├── Conv6 (128→128, 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)
├── Flatten: 128×3×3 = 1152
├── FC1: 1152 → 512 + ReLU + Dropout(0.5)
├── FC2: 512 → 256 + ReLU + Dropout(0.5)
└── FC3: 256 → 10 (출력)
```

### 주요 특징
- **Batch Normalization**: 각 conv 레이어 후 적용으로 학습 안정성 향상
- **작은 Kernel**: 3x3 커널로 파라미터 수 감소하면서 표현력 유지
- **점진적 채널 증가**: 1→32→64→128로 점진적으로 특징 복잡도 증가
- **Data Augmentation**: 회전, 평행이동으로 일반화 성능 향상

## 📊 성능 비교

### 예상 결과
| 메트릭 | 기존 모델 | 개선된 모델 |
|-------|----------|------------|
| 테스트 정확도 | ~98.0% | ~99.5% |
| 파라미터 수 | ~1.2M | ~0.8M |
| 추론 시간 | 빠름 | 약간 느림 |
| 메모리 사용량 | 적음 | 보통 |

### 벤치마크 실행
```bash
# 성능 측정
time ./mnistCUDNN_improved batch

# 메모리 사용량 확인
nvidia-smi
```

## 🐛 문제 해결

### 컴파일 오류
```bash
# CUDA 경로 문제
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# cuDNN 찾을 수 없음
sudo apt-get install libcudnn8-dev

# FreeImage 라이브러리 문제
sudo apt-get install libfreeimage-dev
```

### 런타임 오류
```bash
# 가중치 파일이 없음
ls data_improved/  # 모든 .bin 파일이 있는지 확인

# GPU 메모리 부족
# 더 작은 배치 크기 사용하거나 더 큰 GPU 필요

# 이미지 파일 문제
file test_image/*.pgm  # 올바른 PGM 형식인지 확인
```

## 📁 파일 구조

```
.
├── mnistCUDNN.cpp              # 기존 구현
├── mnistCUDNN_improved.cpp     # 개선된 구현
├── Makefile                    # 기존 Makefile
├── Makefile_improved           # 개선된 Makefile
├── README_improved.md          # 이 파일
├── mnist_trainer/              # 모델 학습 디렉토리
│   ├── improved_mnist_model.py # PyTorch 모델 정의 및 학습
│   ├── requirements.txt        # Python 의존성
│   └── train_and_export.sh     # 학습 스크립트
├── data/                       # 기존 모델 가중치
├── data_improved/              # 개선된 모델 가중치 (학습 후 생성)
├── test_image/                 # 테스트 이미지
└── FreeImage/                  # 이미지 처리 라이브러리
```

## 🔧 커스터마이징

### 하이퍼파라미터 조정
`mnist_trainer/improved_mnist_model.py`에서 다음을 수정할 수 있습니다:
- 학습률: `lr=0.001`
- 배치 크기: `batch_size=128`
- 에포크 수: `num_epochs=20`
- 드롭아웃 비율: `Dropout(0.25)`

### 아키텍처 변경
새로운 레이어를 추가하려면:
1. PyTorch 모델 수정
2. 가중치 내보내기 함수 업데이트
3. CUDNN C++ 코드에 해당 레이어 구현 추가

## 📈 성능 최적화 팁

1. **컴파일 최적화**
   ```bash
   make -f Makefile_improved CXXFLAGS="-O3 -march=native"
   ```

2. **CUDA 아키텍처 최적화**
   ```bash
   # RTX 30xx 시리즈용
   make -f Makefile_improved SMS="86"
   ```

3. **메모리 최적화**
   - 더 작은 배치 크기 사용
   - Mixed precision (FP16) 활용

## 🤝 기여하기

개선사항이나 버그 수정을 원한다면:
1. 이슈 생성
2. 포크 후 브랜치 생성
3. 변경사항 커밋
4. 풀 리퀘스트 생성

## 📝 라이선스

이 프로젝트는 NVIDIA의 원본 샘플 코드를 기반으로 하며, 해당 EULA를 따릅니다.

## 🙏 감사의 말

- NVIDIA의 원본 MNIST CUDNN 샘플
- PyTorch 커뮤니티
- CUDNN 개발팀 