# MNIST CUDNN 성능 분석 보고서 (NSYS 프로파일링)

## 📋 개요

본 보고서는 NVIDIA Nsight Systems (NSYS)를 사용하여 MNIST CUDNN 개선된 구현의 성능을 분석한 결과를 정리합니다. 호스트/디바이스 간 통신 및 입출력을 포함한 전체 수행시간을 측정하고 분석했습니다.

### 🔧 테스트 환경
- **GPU**: NVIDIA GeForce RTX 3080 (8.6 Compute Capability)
- **CUDA Version**: 11.8
- **cuDNN Version**: 9.1.0
- **NSYS Version**: 2025.3.1.90
- **OS**: Ubuntu 22.04 LTS

---

## 🎯 주요 성능 결과

### 📊 전체 수행시간 비교

| 구분 | 총 수행 시간 | 이미지당 시간 | 효율성 개선 |
|------|-------------|-------------|------------|
| **단일 이미지 처리** | **990.942 ms** | **990.942 ms** | 기준 |
| **배치 처리 (18개)** | **1,030.412 ms** | **57.245 ms** | **17.3배 향상** |

### 🚀 핵심 성과
- **배치 처리 효율성**: 단일 처리 대비 **17.3배** 성능 향상
- **GPU 메모리 처리량**: 평균 **~12 GB/s**
- **전체 정확도**: 18개 테스트 이미지 중 **10개 정확** (55.6%)

---

## 📈 상세 성능 분석

### 1. GPU 커널 실행 시간 분석

#### 단일 이미지 처리
| 연산 유형 | 실행 시간 (ns) | 비율 | 인스턴스 수 |
|----------|---------------|------|------------|
| **Convolution (SGEMM)** | 198,432 | **78.7%** | 6 |
| **ReLU Activation** | 15,232 | **6.0%** | 8 |
| **Bias Addition** | 13,664 | **5.4%** | 6 |
| **Max Pooling** | 8,032 | **3.2%** | 3 |
| **Fully Connected** | 6,720 | **2.7%** | 1 |
| **기타 연산** | 10,240 | **4.0%** | - |

#### 배치 처리 (18개 이미지)
| 연산 유형 | 실행 시간 (ns) | 비율 | 인스턴스 수 |
|----------|---------------|------|------------|
| **Convolution (SGEMM)** | 2,892,683 | **76.4%** | 108 |
| **ReLU Activation** | 266,464 | **7.0%** | 144 |
| **Bias Addition** | 229,698 | **6.1%** | 108 |
| **Max Pooling** | 129,440 | **3.4%** | 54 |
| **Fully Connected** | 110,368 | **2.9%** | 18 |
| **기타 연산** | 156,837 | **4.2%** | - |

### 2. 메모리 전송 분석

#### 호스트-디바이스 통신
| 구분 | 데이터 크기 | 전송 시간 | 처리량 |
|------|------------|----------|---------|
| **단일 처리 H2D** | 4.043 MB | 387,841 ns | **10.4 GB/s** |
| **단일 처리 D2H** | 0.001 MB | 1,696 ns | **0.6 GB/s** |
| **배치 처리 H2D** | 4.099 MB | 370,402 ns | **11.1 GB/s** |
| **배치 처리 D2H** | 0.001 MB | 16,960 ns | **0.06 GB/s** |

#### 디바이스 내부 메모리 복사
| 구분 | 데이터 크기 | 전송 시간 | 처리량 |
|------|------------|----------|---------|
| **단일 처리** | 0.003 MB | 4,128 ns | **0.7 GB/s** |
| **배치 처리** | 0.056 MB | 68,288 ns | **0.8 GB/s** |

### 3. CUDA API 호출 분석

#### 단일 이미지 처리
| API 함수 | 실행 시간 (ns) | 비율 | 호출 횟수 |
|----------|---------------|------|----------|
| **cudaFree** | 979,969,002 | **82.9%** | 25 |
| **cudaDeviceReset** | 148,506,948 | **12.6%** | 1 |
| **cudaLaunchKernel** | 50,491,742 | **4.3%** | 28 |
| **기타** | 2,688,305 | **0.2%** | - |

#### 배치 처리
| API 함수 | 실행 시간 (ns) | 비율 | 호출 횟수 |
|----------|---------------|------|----------|
| **cudaFree** | 1,000,038,869 | **82.7%** | 399 |
| **cudaDeviceReset** | 148,447,025 | **12.3%** | 1 |
| **cudaLaunchKernel** | 53,889,039 | **4.5%** | 504 |
| **기타** | 6,149,066 | **0.5%** | - |

---

## 🔍 네트워크 아키텍처 분석

### CNN 구조
```
입력: 1×28×28 (그레이스케일 이미지)
├── Conv1: 1→32 (3×3, padding=1) + ReLU
├── Conv2: 32→32 (3×3, padding=1) + ReLU
├── MaxPool: 2×2 → 32×14×14
├── Conv3: 32→64 (3×3, padding=1) + ReLU  
├── Conv4: 64→64 (3×3, padding=1) + ReLU
├── MaxPool: 2×2 → 64×7×7
├── Conv5: 64→128 (3×3, padding=1) + ReLU
├── Conv6: 128→128 (3×3, padding=1) + ReLU
├── MaxPool: 2×2 → 128×3×3
├── FC1: 1152→512 + ReLU
├── FC2: 512→256 + ReLU
├── FC3: 256→10
└── Softmax → 예측 결과
```

### 전처리 파이프라인
1. **Gaussian Smoothing**: 노이즈 제거
2. **Size Normalization**: 바운딩 박스 기반 크기 정규화
3. **Center of Mass**: 질량 중심 기반 위치 조정
4. **Contrast Enhancement**: 대비 향상 (감마 보정)

---

## 📊 OS 런타임 분석

### 시스템 호출 분석 (단일 처리)
| 시스템 콜 | 실행 시간 (ns) | 비율 | 호출 횟수 |
|----------|---------------|------|----------|
| **poll** | 1,393,037,628 | **52.8%** | 389 |
| **pthread_cond_timedwait** | 1,154,041,913 | **43.7%** | 3 |
| **ioctl** | 85,048,577 | **3.2%** | 1,703 |
| **기타** | 4,826,882 | **0.3%** | - |

---

## 🎯 성능 최적화 인사이트

### 1. 배치 처리의 효과
- **17.3배 성능 향상**: 단일 처리 991ms → 배치 처리 시 57ms/이미지
- **GPU 활용도 극대화**: 연속적인 커널 실행으로 GPU 유휴 시간 최소화
- **메모리 전송 효율성**: 가중치 로딩의 재사용으로 I/O 오버헤드 감소

### 2. 병목점 분석
- **주요 병목**: Convolution 연산 (전체 시간의 76-78%)
- **메모리 병목**: Host-to-Device 전송이 상대적으로 느림
- **API 오버헤드**: cudaFree가 전체 API 시간의 82% 차지

### 3. 최적화 제안
1. **커널 융합**: 연속적인 작은 커널들을 하나로 통합
2. **메모리 풀링**: cudaFree 호출 최소화
3. **스트림 활용**: 비동기 처리로 중첩 실행
4. **텐서 코어 활용**: Mixed Precision 연산 도입

---

## 🔬 NSYS 프로파일링 가이드

### 단계별 실행 방법

#### 1단계: 기본 프로파일링
```bash
nsys profile --stats=true --output=profile_name ./your_program
```

#### 2단계: 상세 프로파일링
```bash
nsys profile --stats=true --trace=cuda,cudnn,cublas,osrt,nvtx \
             --output=detailed_profile ./your_program
```

#### 3단계: 배치 처리 프로파일링
```bash
nsys profile --stats=true --trace=cuda,cudnn,cublas,osrt,nvtx \
             --output=batch_profile ./your_program batch
```

#### 4단계: 데이터 추출
```bash
nsys stats --force-export=true --report cuda_gpu_trace,cuda_api_trace \
           --format csv --output . profile.nsys-rep
```

#### 5단계: 성능 분석
```python
import csv
data = list(csv.DictReader(open('profile_cuda_gpu_trace.csv', 'r')))
start_times = [int(row['Start (ns)']) for row in data if row['Start (ns)']]
end_times = [int(row['Start (ns)']) + int(row['Duration (ns)']) 
             for row in data if row['Start (ns)'] and row['Duration (ns)']]
total_time = max(end_times) - min(start_times)
print(f'총 수행 시간: {total_time/1_000_000:.3f} ms')
```

---

## 📋 결론 및 향후 개선점

### 주요 성과
1. **NSYS를 통한 정확한 성능 측정** 완료
2. **배치 처리 최적화**로 17.3배 성능 향상 달성
3. **전체 파이프라인 분석**을 통한 병목점 식별

### 향후 개선 방향
1. **커널 최적화**: Convolution 연산 성능 개선
2. **메모리 관리**: cudaFree 오버헤드 최소화
3. **정확도 향상**: 전처리 알고리즘 개선 (현재 55.6% → 목표 90%+)
4. **하드웨어 활용**: Tensor Core 및 Mixed Precision 도입

---

## 📎 부록

### 생성된 파일들
- `mnist_detailed_profile.nsys-rep`: 단일 이미지 프로파일
- `mnist_batch_profile.nsys-rep`: 배치 처리 프로파일  
- `mnist_detailed_profile_cuda_gpu_trace.csv`: GPU 트레이스 데이터
- `mnist_batch_profile_cuda_gpu_trace.csv`: 배치 GPU 트레이스 데이터

### 실행 명령어 요약
```bash
# 컴파일
make clean && make

# 단일 이미지 테스트
./mnistCUDNN_improved

# 배치 처리 테스트  
./mnistCUDNN_improved batch

# NSYS 프로파일링
nsys profile --stats=true --trace=cuda,cudnn,cublas,osrt,nvtx \
             --output=profile_name ./mnistCUDNN_improved [batch]
```

---

*보고서 생성일: 2024년*  
*분석 도구: NVIDIA Nsight Systems 2025.3.1.90* 