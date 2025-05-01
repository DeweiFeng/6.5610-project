package utils

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"sort"
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

func StringToInt8(s string) (int8, error) {
	i, err := strconv.Atoi(s)
	if err != nil {
		return 0, err
	}
	if i < -128 || i > 127 {
		return 0, fmt.Errorf("value out of range for int8: %d", i)
	}
	return int8(i), nil
}

func QuantizeClamp(val float64, precBits uint64) int8 {
	scale := 1 << (precBits - 1)
	quantized := int(math.Round(val * float64(scale)))
	return Clamp(quantized, precBits)
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

func FindDBEnd(indices map[uint64]uint, rowStart, colIndex, M, L, maxLen uint64) uint64 {
	rowEnd := rowStart + 1
	for length := uint64(1); ; length++ {
		if (maxLen > 0) && (length >= maxLen) {
			break
		}
		if _, ok := indices[rowEnd*M+colIndex]; ok {
			break
		}
		if rowEnd >= L {
			break
		}
		rowEnd += 1
	}

	return rowEnd
}

func SmoothResults(vals []uint64, mod uint64) []int {
	res := make([]int, len(vals))

	for i := 0; i < len(vals); i++ {
		res[i] = SmoothResult(vals[i], mod)
	}

	return res
}

func SmoothResult(val uint64, mod uint64) int {
	if val > mod {
		panic("Should not happen")
	}

	if val > mod/2 {
		return int(val - mod)
	}

	return int(val)
}

func SortByScores(scores []int) []uint64 {
	score_to_index := make(map[int][]uint64)

	for i, v := range scores {
		if _, ok := score_to_index[v]; !ok {
			score_to_index[v] = make([]uint64, 0)
		}
		score_to_index[v] = append(score_to_index[v], uint64(i))
	}

	sort.Sort(sort.Reverse(sort.IntSlice(scores)))

	indices := make([]uint64, len(scores))
	at := 0
	for i, v := range scores {
		if at >= len(score_to_index[v]) {
			at = 0
		}
		indices[i] = score_to_index[v][at]
		at += 1
	}

	return indices
}

func MessageSizeBytes(m interface{}) uint64 {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	var err error
	switch v := m.(type) {
	// necessary to register the right gob encoders
	case PIR_hint[matrix.Elem32]:
		err = enc.Encode(&v)
	case PIR_hint[matrix.Elem64]:
		err = enc.Encode(&v)
	case pir.Query[matrix.Elem32]:
		err = enc.Encode(&v)
	case pir.Query[matrix.Elem64]:
		err = enc.Encode(&v)
	case pir.Answer[matrix.Elem32]:
		err = enc.Encode(&v)
	case pir.Answer[matrix.Elem64]:
		err = enc.Encode(&v)
	case map[uint]uint64:
		err = enc.Encode(&v)
	case map[uint][]uint64:
		err = enc.Encode(&v)
	case underhood.HintQuery:
		err = enc.Encode(&v)
	case underhood.HintAnswer:
		err = enc.Encode(&v)
	default:
		err = enc.Encode(&v)
		//panic("Bad input to message_size_bytes")
	}

	if err != nil {
		fmt.Println(err)
		panic("Should not happen")
	}

	return uint64(buf.Len())
}

func MessageSizeMB(m interface{}) float64 {
	return BytesToMB(MessageSizeBytes(m))
}

func MessageSizeKB(m interface{}) float64 {
	return BytesToKB(MessageSizeBytes(m))
}

func BytesToMB(bytes uint64) float64 {
	return float64(bytes) / (1024 * 1024)
}

func BytesToKB(bytes uint64) float64 {
	return float64(bytes) / 1024
}
