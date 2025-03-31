package database

import (
	"fmt"
	"os"
	"os/exec"
	"testing"

	"github.com/henrycg/simplepir/rand"
)

func GenerateTestData() {
	// call generate_test_files.py to generate test files

	python_path := "../../.venv/bin/python3"
	script_path := "./generate_test_files.py"
	// create a folder ./test_data
	err := os.MkdirAll("./test_data", os.ModePerm)
	if err != nil {
		panic("Error creating test folder:" + err.Error())
	}
	cmd := exec.Command(python_path, script_path)

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		panic("Error generating test files:" + err.Error())
	}
}

func RemoveTestData() {
	// remove folder ./test_data
	err := os.RemoveAll("./test_data")
	if err != nil {
		panic("Error removing test files:" + err.Error())
	}
}

func TestReadEmbeddingsCsv(t *testing.T) {
	GenerateTestData()
	// Test the ReadEmbeddingsCsv function
	cluster := ReadClusterFromCsv("./test_data/test_cluster_0.csv", 0)

	fmt.Println("Number of vectors:", cluster.NumVectors)
	fmt.Println("Dimension:", cluster.Dim)
	fmt.Println("Precision bits:", cluster.PrecBits)
	fmt.Println("Vectors:")
	for i := 0; i < int(cluster.NumVectors); i++ {
		for j := 0; j < int(cluster.Dim); j++ {
			fmt.Printf("%d ", cluster.Vectors[i*int(cluster.Dim)+j])
		}
		fmt.Println()
	}
	RemoveTestData()
}


func TestReadAllClusters(t *testing.T) {
	GenerateTestData()
	// Test the ReadAllClusters function
	metadata, clusters := ReadAllClusters("./test_data/test")

	fmt.Println("Metadata:")
	fmt.Println("Number of vectors:", metadata.NumVectors)
	fmt.Println("Number of clusters:", metadata.NumClusters)
	fmt.Println("Dimension:", metadata.Dim)
	fmt.Println("Precision bits:", metadata.PrecBits)

	fmt.Println("Clusters:")
	for i, cluster := range clusters {
		fmt.Printf("Cluster %d:\n", i)
		fmt.Println("Number of vectors:", cluster.NumVectors)
		fmt.Println("Dimension:", cluster.Dim)
	}
	RemoveTestData()
}

func TestBuildVectorDatabase(t *testing.T) {
	GenerateTestData()
	// Test the BuildVectorDatabase function
	metadata, clusters := ReadAllClusters("./test_data/test")
	seed := rand.RandomPRGKey()

	// Call BuildVectorDatabase with the clusters
	// hintSz is 900 for text embeddings in Tiptoe, and 500 for image embeddings
	_, _ = BuildVectorDatabase(metadata, clusters, seed, 900)
	RemoveTestData()
}
