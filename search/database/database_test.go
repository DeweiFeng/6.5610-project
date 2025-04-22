package database

import (
	"fmt"
	"testing"

	"github.com/DeweiFeng/6.5610-project/search/utils"
	"github.com/henrycg/simplepir/rand"
)

func TestReadEmbeddingsCsv(t *testing.T) {
	preamble := utils.GenerateTestData()
	// Test the ReadEmbeddingsCsv function
	cluster := ReadClusterFromCsv(preamble+"_cluster_0.csv", 0, 10, 5)

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
	utils.RemoveTestData()
}

func TestReadAllClusters(t *testing.T) {
	preamble := utils.GenerateTestData()
	// Test the ReadAllClusters function
	metadata, clusters := ReadAllClusters(preamble)

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
	utils.RemoveTestData()
}

func TestBuildVectorDatabase(t *testing.T) {
	preamble := utils.GenerateTestData()
	// Test the BuildVectorDatabase function
	metadata, clusters := ReadAllClusters(preamble)
	seed := rand.RandomPRGKey()

	// Call BuildVectorDatabase with the clusters
	// hintSz is 900 for text embeddings in Tiptoe, and 500 for image embeddings
	_, _ = BuildVectorDatabase(metadata, clusters, seed, 900)
	utils.RemoveTestData()
}
