package protocol

import (
	"testing"

	"github.com/DeweiFeng/6.5610-project/search/database"
	"github.com/DeweiFeng/6.5610-project/search/utils"
	"github.com/henrycg/simplepir/matrix"
	"github.com/henrycg/simplepir/pir"
)

func TestZeroQuery(t *testing.T) {
	preamble := utils.GenerateTestData()
	// Test the BuildVectorDatabase function
	metadata, clusters := database.ReadAllClusters(preamble)

	hinSz := uint64(900) // hintSz is 900 for text embeddings in Tiptoe, and 500 for image embeddings
	// get an empty server
	s := new(Server)
	s.ProcessVectorsFromClusters(metadata, clusters, hinSz)

	c := new(Client)
	c.Setup(s.Hint) // get the hint from the server

	hintServer := SetUpHintServer(s.Hint)

	// from here, it is for each query
	ct := c.PreprocessQuery()
	offlineAns := hintServer.HintAnswer(ct)
	c.ProcessHintApply(offlineAns)

	// get the query: a list of uint64 of zeros, size = dim
	zeroQuery := make([]int8, metadata.Dim)
	for i := 0; i < int(metadata.Dim); i++ {
		zeroQuery[i] = 0
	}
	clusterIndex := uint64(0)

	query := c.QueryEmbeddings(zeroQuery, clusterIndex)

	var ans pir.Answer[matrix.Elem64]
	s.Answer(query, &ans)

	dec := c.ReconstructEmbeddingsWithinCluster(&ans, clusterIndex)
	scores := utils.SmoothResults(dec, c.DBInfo.P())

	// check if all scores are zero
	for i := 0; i < len(scores); i++ {
		if scores[i] != 0 {
			t.Errorf("Expected score %d to be 0, but got %d", i, scores[i])
		}
	}

	// print all scores one by one, in one line
	print("Scores: ")
	for i := 0; i < len(scores); i++ {
		if i == len(scores)-1 {
			print(scores[i])
		} else {
			print(scores[i], ", ")
		}
	}
	// print a new line
	println()

	utils.RemoveTestData()
}
