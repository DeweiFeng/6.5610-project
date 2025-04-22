package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/DeweiFeng/6.5610-project/search/database"
	"github.com/DeweiFeng/6.5610-project/search/protocol"
	"github.com/DeweiFeng/6.5610-project/search/utils"
)

func argumentsValidation(preamble string, topk int) {
	if preamble == "" {
		panic("Error: Preamble is required")
	}
	if topk <= 0 {
		panic("Error: topk must be a positive integer")
	}
}

func readQueryLine(reader *csv.Reader, dim uint64) (uint64, []int8, error) {
	row, err := reader.Read()
	if err != nil {
		return 0, nil, err
	}
	if len(row) != int(dim)+1 {
		return 0, nil, fmt.Errorf("expected %d columns, got %d", dim+1, len(row))
	}
	clusterIndex, err := utils.StringToUint64(row[0])
	if err != nil {
		panic("Error converting cluster index to uint64: " + err.Error())
	}
	query := make([]int8, dim)
	for i := 0; i < int(dim); i++ {
		query[i], err = utils.StringToInt8(row[i+1])
		if err != nil {
			panic("Error converting query to int8: " + err.Error())
		}
	}
	return clusterIndex, query, nil
}

func writeResults(writer *csv.Writer, clusterIndex uint64, indices []uint64) {
	line := make([]string, len(indices)*2)
	for i, index := range indices {
		line[i*2] = fmt.Sprintf("%d", clusterIndex)
		line[i*2+1] = fmt.Sprintf("%d", index)
	}
	if err := writer.Write(line); err != nil {
		panic("Error writing to output file: " + err.Error())
	}
	writer.Flush()
}

func main() {
	preamble := flag.String("preamble", "", "Preamble to use for the search")
	topk := flag.Int("topk", 10, "Number of top results to return")

	flag.Parse()
	argumentsValidation(*preamble, *topk)

	metadata, clusters := database.ReadAllClusters(*preamble)
	hintSz := uint64(900)

	server := new(protocol.Server)
	server.ProcessVectorsFromClusters(metadata, clusters, hintSz)

	client := new(protocol.Client)
	client.Setup(server.Hint)

	dir := filepath.Dir(*preamble)
	prefix := filepath.Base(*preamble)
	queryFile := utils.OpenFile(filepath.Join(dir, prefix+"_queries.csv"))
	defer queryFile.Close()

	reader := csv.NewReader(queryFile)

	outputFile, err := os.Create(filepath.Join(dir, prefix+"_results.csv"))
	if err != nil {
		panic("Error creating output file: " + err.Error())
	}
	defer outputFile.Close()
	writer := csv.NewWriter(outputFile)
	defer writer.Flush()

	for {
		ct := client.PreprocessQuery()
		offlineAns := server.HintAnswer(ct)
		client.ProcessHintApply(offlineAns)

		clusterIndex, query, err := readQueryLine(reader, metadata.Dim)

		if err == io.EOF {
			break
		}
		if err != nil {
			panic("Error reading query line: " + err.Error())
		}

		fmt.Printf("Processing query %v of cluster %d\n", query, clusterIndex)
		indices := runRound(client, server, query, clusterIndex)
		if len(indices) > *topk {
			indices = indices[:*topk]
		}
		writeResults(writer, clusterIndex, indices)
	}
}

func runRound(c *protocol.Client, s *protocol.Server, query []int8, clusterIndex uint64) []uint64 {
	queryEmb := c.QueryEmbeddings(query, clusterIndex)
	ans := s.Answer(queryEmb)

	dec := c.ReconstructEmbeddingsWithinCluster(ans, clusterIndex)
	scores := utils.SmoothResults(dec, c.DBInfo.P())

	indices := utils.SortByScores(scores)
	return indices
}
