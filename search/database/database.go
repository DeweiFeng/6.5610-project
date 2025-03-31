package database

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/DeweiFeng/6.5610-project/search/utils"
	"github.com/henrycg/simplepir/lwe"
	"github.com/henrycg/simplepir/matrix"
	"github.com/henrycg/simplepir/pir"
	"github.com/henrycg/simplepir/rand"
)

type ClusterMap map[uint]uint64

// DBIndex calculates the index of an element in the database
func DBIndex(row, col, m uint64) uint64 {
	return row*m + col
}

type Metadata struct {
	NumVectors  uint64 `json:"num_vectors"`
	Dim         uint64 `json:"dim"`
	NumClusters uint64 `json:"num_clusters"`
	PrecBits    uint64 `json:"prec_bits"`
}

type Cluster struct {
	Index      uint64
	NumVectors uint64
	Dim        uint64
	PrecBits   uint64
	Vectors    []int8
}

func ReadClusterFromCsv(file string, index uint64) *Cluster {
	f, err := os.Open(file)
	if err != nil {
		fmt.Println(err)
		panic("Error opening file " + file)
	}
	defer f.Close()

	reader := csv.NewReader(f)

	line1, err1 := reader.Read()
	line2, err2 := reader.Read()
	line3, err3 := reader.Read()
	if err1 != nil || err2 != nil || err3 != nil {
		panic("Error reading headers of CSV file " + file)
	}

	numVec, err1 := utils.StringToUint64(line1[0])
	dim, err2 := utils.StringToUint64(line2[0])
	precBits, err3 := utils.StringToUint64(line3[0])
	if err1 != nil || err2 != nil || err3 != nil {
		panic("Error parsing headers of CSV file " + file)
	}

	vectors := make([]int8, numVec*dim)

	reader.FieldsPerRecord = int(dim)

	for at := uint64(0); ; at++ {
		row, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Printf("%s\n", err)
			panic("Error reading CSV file " + file)
		}

		offset := at * dim

		for i := uint64(0); i < dim; i++ {
			u, err := strconv.Atoi(row[i])
			vectors[offset+i] = utils.Clamp(u, precBits)
			if err != nil {
				panic("Error parsing CSV embeddings" + file)
			}
		}
	}

	return &Cluster{
		Index:      index,
		NumVectors: numVec,
		Dim:        dim,
		PrecBits:   precBits,
		Vectors:    vectors,
	}
}

func PackClusters(clusters []*Cluster, maxCapacity uint64) ([][]uint, []uint64) {
	numClusters := uint64(len(clusters))
	if numClusters == 0 {
		panic("No clusters given")
	}
	clusterIndices := make([]uint64, numClusters)

	for i := uint64(0); i < numClusters; i++ {
		clusterIndices[i] = i
	}

	// reverse sort clusterIndices by their size, largest first
	sort.Slice(clusterIndices, func(i, j int) bool {
		return clusters[clusterIndices[i]].NumVectors > clusters[clusterIndices[j]].NumVectors
	})

	fmt.Printf("The longest row has length %d -- max capacity is %d\n", clusters[0].NumVectors, maxCapacity)

	if clusters[clusterIndices[0]].NumVectors > maxCapacity {
		maxCapacity = clusters[clusterIndices[0]].NumVectors
	}

	cols := make([][]uint, 1)
	cols[0] = []uint{uint(clusters[clusterIndices[0]].Index)}
	col_szs := []uint64{clusters[clusterIndices[0]].NumVectors}

	for i := uint64(1); i < numClusters; i++ {
		fit := false
		for j := 0; j < len(cols); j++ {
			if col_szs[j]+clusters[clusterIndices[i]].NumVectors < maxCapacity {
				col_szs[j] += clusters[clusterIndices[i]].NumVectors
				cols[j] = append(cols[j], uint(clusters[clusterIndices[i]].Index))
				fit = true
				break
			}
		}

		if !fit {
			new_col := []uint{uint(clusters[clusterIndices[i]].Index)}
			cols = append(cols, new_col)
			col_szs = append(col_szs, clusters[clusterIndices[i]].NumVectors)
		}
	}

	return cols, col_szs
}

func ReadAllClusters(clusterPreamble string) (Metadata, []*Cluster) {
	dir := filepath.Dir(clusterPreamble)
	prefix := filepath.Base(clusterPreamble)

	jsonFile := utils.OpenFile(filepath.Join(dir, prefix+"_metadata.json"))
	defer jsonFile.Close()

	decoder := json.NewDecoder(jsonFile)
	var metadata Metadata
	if err := decoder.Decode(&metadata); err != nil {
		panic("Error decoding metadata file")
	}

	numVectors := metadata.NumVectors
	numClusters := metadata.NumClusters
	dim := metadata.Dim
	precBits := metadata.PrecBits

	// file names of clusters are dir/prefix_cluster_0.csv, ..., until the last cluster (number of clusters is metadata.NumClusters)

	fmt.Printf("Building database with %d %d-dim %d-bit vectors, organized in %d clusters\n", numVectors, dim, precBits, numClusters)

	// call ReadEmbeddingsCsv for each cluster, to get a slice of clusters
	// clusters := make([]*Cluster, numClusters)
	cluster_sizes := make([]uint64, numClusters)

	vecCountVeri := uint64(0)

	clusters := make([]*Cluster, numClusters)

	for i := uint64(0); i < numClusters; i++ {
		clusterFile := filepath.Join(dir, fmt.Sprintf("%s_cluster_%d.csv", prefix, i))
		// clusterNumVec, clusterDim, clusterPrecBits, clusterVec := ReadClusterFromCsv(clusterFile)
		clusters[i] = ReadClusterFromCsv(clusterFile, i)
		cluster_sizes[i] = clusters[i].NumVectors
		vecCountVeri += clusters[i].NumVectors

		if clusters[i].Dim != dim {
			panic("Dimension mismatch")
		}
		if clusters[i].PrecBits != precBits {
			panic("Precision mismatch")
		}
	}

	if vecCountVeri != numVectors {
		panic("Total number of vectors mismatch")
	}

	return metadata, clusters
}

// BuildVectorDatabase creates a PIR database from CSV vector files
func BuildVectorDatabase(metadata Metadata, clusters []*Cluster, seed *rand.PRGKey, hintSz uint64) (*pir.Database[matrix.Elem64], ClusterMap) {

	numVectors := metadata.NumVectors
	dim := metadata.Dim
	precBits := metadata.PrecBits

	l := hintSz * 125
	logQ := uint64(64)

	actualSz := uint64(numVectors * dim) // total number of values
	cols, colSzs := PackClusters(clusters, l)

	m := uint64(len(cols)) * dim
	l = utils.Max(colSzs)
	fmt.Printf("DB size is %d -- best possible would be %d\n", l*m, actualSz)

	// Pick SimplePIR params
	recordLen := 15 // in tiptoe, it's 15 for text embeddings and 17 for image embeddings
	p := lwe.NewParamsFixedP(logQ, m, (1 << recordLen))
	if (p == nil) || (p.P < uint64(1<<precBits)) || (p.Logq != 64) {
		fmt.Printf("P = %d; LogQ = %d\n", p.P, p.Logq)
		panic("Failure in picking SimplePIR DB parameters")
	}

	// Store embddings in database, such that clusters are kept together in a column
	vals := make([]uint64, l*m)
	indexMap := make(map[uint]uint64)
	slots := dim

	for colIndex, colContents := range cols {
		rowIndex := uint64(0)
		for _, clusterIndex := range colContents {
			if _, ok := indexMap[clusterIndex]; ok {
				panic("Key should not yet exist")
			}

			indexMap[clusterIndex] = DBIndex(rowIndex, slots*uint64(colIndex), m)

			sz := clusters[clusterIndex].NumVectors
			start := uint64(0)

			for x := uint64(0); x < sz; x++ {
				for j := uint64(0); j < slots; j++ {
					vals[DBIndex(rowIndex, slots*uint64(colIndex)+j, m)] = uint64(clusters[clusterIndex].Vectors[start+j])
				}
				start += slots
				rowIndex += 1
				if rowIndex > l {
					panic("Should not happen")
				}
			}
		}
	}

	db := pir.NewDatabaseFixedParams[matrix.Elem64](l*m, uint64(recordLen), vals, p)
	fmt.Printf("DB dimensions: %d by %d\n", db.Info.L, db.Info.M)

	if db.Info.L != l {
		panic("Should not happen")
	}

	return db, indexMap
}
