package protocol

import (
	"sort"

	"github.com/DeweiFeng/6.5610-project/search/database"
	"github.com/DeweiFeng/6.5610-project/search/utils"
	"github.com/ahenzinger/underhood/underhood"
	"github.com/henrycg/simplepir/matrix"
	"github.com/henrycg/simplepir/pir"
)

type QueryType interface {
	bool | underhood.HintQuery | pir.Query[matrix.Elem64] | pir.Query[matrix.Elem32]
}

type AnsType interface {
	TiptoeHint | underhood.HintAnswer | pir.Answer[matrix.Elem64] | pir.Answer[matrix.Elem32]
}

type Client struct {
	Metadata        database.Metadata
	UnderhoodClient *underhood.Client[matrix.Elem64]

	DBInfo         *pir.DBInfo
	ClusterToIndex database.ClusterMap
	IndexToCluster map[uint64]uint
}

func (c *Client) Free() {
	c.UnderhoodClient.Free()
}

func (c *Client) Setup(hint *TiptoeHint) {
	c.Metadata = hint.Metadata
	c.DBInfo = &hint.PIRHint.Info
	c.ClusterToIndex = hint.IndexMap
	c.UnderhoodClient = utils.NewUnderhoodClient(&hint.PIRHint)
	// c.Indices = make(map[uint64]bool) // is this index (of DB) a start of a cluster?
	c.IndexToCluster = make(map[uint64]uint)
	for k, v := range c.ClusterToIndex {
		c.IndexToCluster[v] = k
	}
}

func (c *Client) PreprocessQuery() *underhood.HintQuery {
	return c.UnderhoodClient.HintQuery()
}

func (c *Client) ProcessHintApply(ans *underhood.HintAnswer) {
	c.UnderhoodClient.HintRecover(ans)
	c.UnderhoodClient.PreprocessQueryLHE()
}

func (c *Client) QueryEmbeddings(emb []int8, clusterIndex uint64) *pir.Query[matrix.Elem64] {
	// check if the clusterIndex is valid
	if clusterIndex >= uint64(len(c.ClusterToIndex)) {
		panic("Invalid cluster index")
	}

	dbIndex := c.ClusterToIndex[uint(clusterIndex)]
	m := c.DBInfo.M
	dim := uint64(len(emb))

	if m%dim != 0 {
		panic("Should not happen")
	}
	if dbIndex%dim != 0 {
		panic("Should not happen")
	}

	colIndex := dbIndex % m
	arr := matrix.Zeros[matrix.Elem64](m, 1)
	for j := uint64(0); j < dim; j++ {
		arr.AddAt(colIndex+j, 0, matrix.Elem64(emb[j]))
	}

	return c.UnderhoodClient.QueryLHE(arr)
}

func (c *Client) ReconstructWithinCluster(answer *pir.Answer[matrix.Elem64], clusterIndex uint64, mod uint64) *[]VectorScore {
	dbIndex := c.ClusterToIndex[uint(clusterIndex)]
	rowStart := dbIndex / c.DBInfo.M
	colIndex := dbIndex % c.DBInfo.M
	rowEnd := utils.FindDBEnd(c.IndexToCluster, rowStart, colIndex, c.DBInfo.M, c.DBInfo.L, 0)

	vals := c.UnderhoodClient.RecoverLHE(answer)

	res := make([]VectorScore, rowEnd-rowStart)
	at := 0
	for j := rowStart; j < rowEnd; j++ {
		// res[at] = uint64(vals.Get(j, 0))
		res[at] = VectorScore{
			ClusterID:       uint(clusterIndex),
			IDWithinCluster: uint64(at),
			Score:           utils.SmoothResult(uint64(vals.Get(j, 0)), mod),
		}
		at += 1
	}

	sort.Slice(res, func(i, j int) bool {
		return res[i].Score > res[j].Score
	})

	return &res
}

// define a struct that saves cluster id, id within cluster, and value
type VectorScore struct {
	ClusterID       uint
	IDWithinCluster uint64
	Score           int
}

func (c *Client) ReconstructWithinBin(answer *pir.Answer[matrix.Elem64], clusterIndex uint64, mod uint64) *[]VectorScore {
	vals := c.UnderhoodClient.RecoverLHE(answer)
	res := make([]VectorScore, 0)
	colIndex := c.ClusterToIndex[uint(clusterIndex)] % c.DBInfo.M

	currStart := uint64(0)
	for {
		currCluster, ok := c.IndexToCluster[currStart*c.DBInfo.M+colIndex]
		if !ok {
			break
		}
		currEnd := utils.FindDBEnd(c.IndexToCluster, currStart, colIndex, c.DBInfo.M, c.DBInfo.L, 0)

		at := 0
		for j := currStart; j < currEnd; j++ {
			res = append(res, VectorScore{
				ClusterID:       currCluster,
				IDWithinCluster: uint64(at),
				Score:           utils.SmoothResult(uint64(vals.Get(j, 0)), mod),
			})
			at += 1
		}
		currStart = currEnd
	}

	sort.Slice(res, func(i, j int) bool {
		return res[i].Score > res[j].Score
	})

	return &res
}
