package protocol

import (
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

	DBInfo   *pir.DBInfo
	IndexMap database.ClusterMap
	Indices  map[uint64]bool
}

func (c *Client) Free() {
	c.UnderhoodClient.Free()
}

func (c *Client) Setup(hint *TiptoeHint) {
	c.Metadata = hint.Metadata
	c.DBInfo = &hint.PIRHint.Info
	c.IndexMap = hint.IndexMap
	c.UnderhoodClient = utils.NewUnderhoodClient(&hint.PIRHint)
	c.Indices = make(map[uint64]bool)
	for _, v := range c.IndexMap {
		c.Indices[v] = true
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
	if clusterIndex >= uint64(len(c.IndexMap)) {
		panic("Invalid cluster index")
	}

	dbIndex := c.IndexMap[uint(clusterIndex)]
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

func (c *Client) ReconstructEmbeddings(answer *pir.Answer[matrix.Elem64], clusterIndex uint64) uint64 {
	vals := c.UnderhoodClient.RecoverLHE(answer)

	dbIndex := c.IndexMap[uint(clusterIndex)]
	rowIndex := dbIndex / c.DBInfo.M
	res := vals.Get(rowIndex, 0)

	return uint64(res)
}

func (c *Client) ReconstructEmbeddingsWithinCluster(answer *pir.Answer[matrix.Elem64], clusterIndex uint64) []uint64 {
	dbIndex := c.IndexMap[uint(clusterIndex)]
	rowStart := dbIndex / c.DBInfo.M
	colIndex := dbIndex % c.DBInfo.M
	rowEnd := utils.FindDBEnd(c.Indices, rowStart, colIndex, c.DBInfo.M, c.DBInfo.L, 0)

	vals := c.UnderhoodClient.RecoverLHE(answer)

	res := make([]uint64, rowEnd-rowStart)
	at := 0
	for j := rowStart; j < rowEnd; j++ {
		res[at] = uint64(vals.Get(j, 0))
		at += 1
	}

	return res
}
