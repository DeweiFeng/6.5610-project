package database

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/henrycg/simplepir/lwe"
	"github.com/henrycg/simplepir/matrix"
	"github.com/henrycg/simplepir/pir"
	"github.com/henrycg/simplepir/rand"
)

// ClusterMap maps cluster IDs to database indices
type ClusterMap map[uint]uint64

// DBIndex calculates the index of an element in the database
func DBIndex(row, col, m uint64) uint64 {
	return row*m + col
}

// LoadVectorsFromCSV reads integer vectors from a CSV file
func LoadVectorsFromCSV(path string) ([][]uint64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open CSV file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var vectors [][]uint64

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error reading CSV record: %v", err)
		}

		// Convert string values to uint64
		vector := make([]uint64, len(record))
		for i, val := range record {
			num, err := strconv.ParseUint(val, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("error parsing integer at row %d, col %d: %v", len(vectors)+1, i+1, err)
			}
			vector[i] = num
		}
		vectors = append(vectors, vector)
	}

	return vectors, nil
}

// BuildVectorDatabase creates a PIR database from CSV vector files
func BuildVectorDatabase(clusterPreamble string, vectorDim uint64, seed *rand.PRGKey, hintSz uint64) (*pir.Database[matrix.Elem64], ClusterMap, error) {
	dir := filepath.Dir(clusterPreamble)
	prefix := filepath.Base(clusterPreamble)

	// Find all files in directory that match the prefix and have .csv extension
	dirEntries, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read directory %s: %v", dir, err)
	}

	var clusterFiles []string
	for _, entry := range dirEntries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), prefix) && strings.HasSuffix(entry.Name(), ".csv") {
			clusterFiles = append(clusterFiles, filepath.Join(dir, entry.Name()))
		}
	}

	// Count clusters
	clusterCount := uint64(len(clusterFiles))
	if clusterCount == 0 {
		return nil, nil, fmt.Errorf("no cluster files provided")
	}

	fmt.Printf("Building database with %d clusters\n", clusterCount)

	// Load all vectors to determine dimensions and validate
	allClusters := make([][][]uint64, clusterCount)
	maxVectorsPerCluster := uint64(0)

	for i, filePath := range clusterFiles {
		vectors, err := LoadVectorsFromCSV(filePath)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to load vectors from %s: %v", filePath, err)
		}

		if len(vectors) == 0 {
			return nil, nil, fmt.Errorf("cluster file %s contains no vectors", filePath)
		}

		// Verify vector dimension
		for j, vec := range vectors {
			if uint64(len(vec)) != vectorDim {
				return nil, nil, fmt.Errorf("dimension mismatch in file %s, vector %d: expected %d, got %d",
					filePath, j, vectorDim, len(vec))
			}
		}

		allClusters[i] = vectors

		// Track maximum vectors per cluster for padding
		if uint64(len(vectors)) > maxVectorsPerCluster {
			maxVectorsPerCluster = uint64(len(vectors))
		}
	}

	// Set database parameters
	l := maxVectorsPerCluster
	if hintSz > 0 {
		suggestedL := hintSz * 125
		if suggestedL > l {
			fmt.Printf("Increasing l from %d to %d based on hint size\n", l, suggestedL)
			l = suggestedL
		}
	}

	// m = number of clusters * vector dimension
	m := clusterCount * vectorDim
	logQ := uint64(64)
	recordLen := uint64(16) // Assuming each element uses 16 bits

	fmt.Printf("DB dimensions: l=%d, m=%d\n", l, m)

	// Pick SimplePIR params
	p := lwe.NewParamsFixedP(logQ, m, (1 << recordLen))
	if p == nil || p.Logq != 64 {
		return nil, nil, fmt.Errorf("failure in picking SimplePIR DB parameters: P=%d, LogQ=%d", p.P, p.Logq)
	}

	// Create database array and map for cluster indices
	vals := make([]uint64, l*m)
	indexMap := make(ClusterMap)

	// Fill database with vectors
	for clusterIdx, vectors := range allClusters {
		// Store starting index for this cluster
		clusterStartCol := uint64(clusterIdx) * vectorDim
		indexMap[uint(clusterIdx)] = DBIndex(0, clusterStartCol, m)

		// Store vectors in database
		for vecIdx, vec := range vectors {
			rowIdx := uint64(vecIdx)

			// Store each dimension of the vector
			for dimIdx := uint64(0); dimIdx < vectorDim; dimIdx++ {
				colIdx := clusterStartCol + dimIdx
				vals[DBIndex(rowIdx, colIdx, m)] = vec[dimIdx]
			}
		}

		// Pad with zeros if needed
		for vecIdx := uint64(len(vectors)); vecIdx < l; vecIdx++ {
			for dimIdx := uint64(0); dimIdx < vectorDim; dimIdx++ {
				colIdx := clusterStartCol + dimIdx
				vals[DBIndex(vecIdx, colIdx, m)] = 0
			}
		}
	}

	// Create database
	db := pir.NewDatabaseFixedParams[matrix.Elem64](l*m, recordLen, vals, p)

	if db.Info.L != l {
		return nil, nil, fmt.Errorf("database creation error: expected L=%d, got L=%d", l, db.Info.L)
	}

	fmt.Printf("Created database with dimensions L=%d, M=%d\n", db.Info.L, db.Info.M)

	return db, indexMap, nil
}
