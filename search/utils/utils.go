package utils

import (
	"fmt"
	"os"
	"strconv"
)

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
