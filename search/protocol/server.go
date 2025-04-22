package protocol

import (
	"fmt"

	"github.com/DeweiFeng/6.5610-project/search/database"
	"github.com/DeweiFeng/6.5610-project/search/utils"
	"github.com/ahenzinger/underhood/underhood"
	"github.com/henrycg/simplepir/matrix"
	"github.com/henrycg/simplepir/pir"
	"github.com/henrycg/simplepir/rand"
)

type TiptoeHint struct {
	Metadata database.Metadata

	PIRHint  utils.PIR_hint[matrix.Elem64]
	IndexMap database.ClusterMap
}

type Server struct {
	Hint       *TiptoeHint
	PIRServer  *pir.Server[matrix.Elem64]
	HintServer *underhood.Server[matrix.Elem64]
}

func (s *Server) ProcessVectorsFromClusters(metadata database.Metadata, clusters []*database.Cluster, hintSz uint64, precBits uint64) {
	seed := rand.RandomPRGKey()

	numClusters := metadata.NumClusters
	dim := metadata.Dim
	numVectors := metadata.NumVectors

	fmt.Printf("Preprocessing of %d %d-dim %d-bit embeddings organized in %d clusters\n", numVectors, dim, precBits, numClusters)

	db, indexMap := database.BuildVectorDatabase(metadata, clusters, seed, hintSz, precBits)
	s.PIRServer = pir.NewServerSeed(db, seed)

	s.Hint = new(TiptoeHint)
	s.Hint.Metadata = metadata

	s.Hint.PIRHint.Hint = *s.PIRServer.Hint()
	s.Hint.PIRHint.Info = *s.PIRServer.DBInfo()
	s.Hint.PIRHint.Seeds = []rand.PRGKey{*seed}
	s.Hint.PIRHint.Offsets = []uint64{s.Hint.PIRHint.Info.M}
	s.Hint.IndexMap = indexMap

	s.HintServer = underhood.NewServerHintOnly(&s.Hint.PIRHint.Hint)

	// THIS CHECK DOES NOT MAKE SENSE FOR IMAGE DATASET, BECAUSE VECTORS ARE NORMALIZED
	max_inner_prod := 2 * (1 << (2*precBits - 2)) * dim
	if s.PIRServer.Params().P < max_inner_prod {
		fmt.Printf("%d < %d\n", s.PIRServer.Params().P, max_inner_prod)
		panic("Parameters not supported. Inner products may wrap around.")
	}
}

func (s *Server) HintAnswer(ct *[][]byte) *underhood.HintAnswer {
	offlineAns := s.HintServer.HintAnswer(ct)
	return offlineAns
}

func (s *Server) Answer(query *pir.Query[matrix.Elem64]) *pir.Answer[matrix.Elem64] {
	ans := s.PIRServer.Answer(query)
	return ans
}
