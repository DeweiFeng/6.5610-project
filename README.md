# 6.5610-project

Dependencies: C compiler (like GCC), Go 1.20.2, SEAL compiled with `-DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=off` and `-DSEAL_USE_INTEL_HEXL=ON` and Python 3.

Install Go dependencies:
```bash
go mod tidy
```

Usage (with test data):
```bash
go run main.go --preamble=test_data/test --topk=10
```
