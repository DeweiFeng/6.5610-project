package protocol

import (
	"testing"

	"github.com/DeweiFeng/6.5610-project/search/database"
	"github.com/DeweiFeng/6.5610-project/search/utils"
)

func TestProcessVectorsFromClusters(t *testing.T) {
	preamble := utils.GenerateTestData()
	// Test the BuildVectorDatabase function
	metadata, clusters := database.ReadAllClusters(preamble, 5)

	hintSz := uint64(900) // hintSz is 900 for text embeddings in Tiptoe, and 500 for image embeddings
	// get an empty server
	s := new(Server)
	s.ProcessVectorsFromClusters(metadata, clusters, hintSz, 5)
	utils.RemoveTestData()
}
