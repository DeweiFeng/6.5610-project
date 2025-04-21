package utils

import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strconv"

	"github.com/ahenzinger/underhood/underhood"
	"github.com/henrycg/simplepir/matrix"
	"github.com/henrycg/simplepir/pir"
	"github.com/henrycg/simplepir/rand"
)

type PIR_hint[T matrix.Elem] struct {
	Info pir.DBInfo
	Hint matrix.Matrix[T]

	Seeds   []rand.PRGKey
	Offsets []uint64
}

func StringToUint(s string) (uint, error) {
	i, err := strconv.Atoi(s)
	return uint(i), err
}

func StringToUint64(s string) (uint64, error) {
	i, err := strconv.ParseUint(s, 10, 64)
	return i, err
}

func Clamp(val int, precBits uint64) int8 {
	min := -int(1 << (precBits - 1))
	if val <= min {
		return int8(min)
	}

	max := int(1 << (precBits - 1))
	if val > max {
		return int8(max)
	}

	return int8(val)
}

func OpenFile(file string) *os.File {
	f, err := os.Open(file)
	if err != nil {
		fmt.Println(err)
		panic("Error opening file")
	}
	return f
}

func Max(arr []uint64) uint64 {
	res := uint64(0)
	for _, v := range arr {
		if v > res {
			res = v
		}
	}
	return res
}

func GenerateTestData() string {
	// call generate_test_files.py to generate test files

	// get the path of the current file
	current_path := GetCurrentFileDirectory()

	python_path := path.Join(current_path, "../../.venv/bin/python3")
	python_path, err := filepath.Abs(python_path)
	if err != nil {
		panic("Error getting python path:" + err.Error())
	}
	script_path := path.Join(current_path, "./generate_test_files.py")
	script_path, err = filepath.Abs(script_path)
	if err != nil {
		panic("Error getting script path:" + err.Error())
	}

	data_path := path.Join(current_path, "../test_data")
	data_path, err = filepath.Abs(data_path)
	if err != nil {
		panic("Error getting data path:" + err.Error())
	}

	// create a folder ./test_data
	err = os.MkdirAll(data_path, os.ModePerm)
	if err != nil {
		panic("Error creating test folder:" + err.Error())
	}
	// script takes four arguments: num_vectors, dim, num_clusters, preamble
	num_vectors := 100
	dim := 10
	num_clusters := 5
	preamble := path.Join(data_path, "test")
	// run the script
	cmd := exec.Command(python_path, script_path, strconv.Itoa(num_vectors), strconv.Itoa(dim), strconv.Itoa(num_clusters), preamble)

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		panic("Error generating test files:" + err.Error())
	}

	return preamble
}

func RemoveTestData() {
	current_path := GetCurrentFileDirectory()
	data_path := path.Join(current_path, "../test_data")
	err := os.RemoveAll(data_path)
	if err != nil {
		panic("Error removing test files:" + err.Error())
	}
}

// GetCurrentFileDirectory returns the directory of utils.go
func GetCurrentFileDirectory() string {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		dir, err := os.Getwd()
		if err != nil {
			return "."
		}
		return dir
	}

	return filepath.Dir(filename)
}

// GetCallerDirectory returns the directory of the file that called this function
func GetCallerDirectory() string {
	_, filename, _, ok := runtime.Caller(1) // Using 1 instead of 0 to get the caller's info
	if !ok {
		dir, err := os.Getwd()
		if err != nil {
			return "."
		}
		return dir
	}

	return filepath.Dir(filename)
}

func NewUnderhoodClient[T matrix.Elem](h *PIR_hint[T]) *underhood.Client[T] {
	return underhood.NewClientDistributed[T](h.Seeds, h.Offsets, &h.Info)
}

func FindDBEnd(indices map[uint64]bool, rowStart, colIndex, M, L, maxLen uint64) uint64 {
	rowEnd := rowStart + 1
	for length := uint64(1); ; length++ {
	  if (maxLen > 0) && (length >= maxLen) {
		break
	  }
	  if _, ok := indices[rowEnd * M + colIndex]; ok {
		break
	  }
	  if rowEnd >= L {
		break
	  }
	  rowEnd += 1
	}
  
	return rowEnd
  }