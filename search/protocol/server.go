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
	Hint      *TiptoeHint
	PIRServer *pir.Server[matrix.Elem64]
}

func (s *Server) ProcessVectorsFromClusters(metadata database.Metadata, clusters []*database.Cluster, hintSz uint64) {
	seed := rand.RandomPRGKey()

	numClusters := metadata.NumClusters
	dim := metadata.Dim
	numVectors := metadata.NumVectors
	precBits := metadata.PrecBits

	fmt.Printf("Preprocessing of %d %d-dim %d-bit embeddings organized in %d clusters\n", numVectors, dim, precBits, numClusters)

	db, indexMap := database.BuildVectorDatabase(metadata, clusters, seed, hintSz)
	s.PIRServer = pir.NewServerSeed(db, seed)

	s.Hint = new(TiptoeHint)
	s.Hint.Metadata = metadata

	s.Hint.PIRHint.Hint = *s.PIRServer.Hint()
	s.Hint.PIRHint.Info = *s.PIRServer.DBInfo()
	s.Hint.PIRHint.Seeds = []rand.PRGKey{*seed}
	s.Hint.PIRHint.Offsets = []uint64{s.Hint.PIRHint.Info.M}
	s.Hint.IndexMap = indexMap

	// THIS CHECK DOES NOT MAKE SENSE FOR IMAGE DATASET, BECAUSE VECTORS ARE NORMALIZED
	max_inner_prod := 2 * (1 << (2*precBits - 2)) * dim
	if s.PIRServer.Params().P < max_inner_prod {
		fmt.Printf("%d < %d\n", s.PIRServer.Params().P, max_inner_prod)
		panic("Parameters not supported. Inner products may wrap around.")
	}

	fmt.Println("    done")
}

func (s *Server) Answer(query *pir.Query[matrix.Elem64], ans *pir.Answer[matrix.Elem64]) error {
	*ans = *s.PIRServer.Answer(query)
	return nil
}

func SetUpHintServer(h *TiptoeHint) *underhood.Server[matrix.Elem64] {
	out := new(underhood.Server[matrix.Elem64])
	if h.PIRHint.Hint.Cols() != 0 {
		out = underhood.NewServerHintOnly(&h.PIRHint.Hint)
	}
	return out
}
