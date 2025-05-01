package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"time"

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

func readQueryLine(reader *csv.Reader, dim uint64, precBits uint64) (uint64, []int8, bool) {
	row, err := reader.Read()
	if err == io.EOF {
		return 0, nil, true
	}
	if err != nil {
		panic("Error reading query line: " + err.Error())
	}
	if len(row) != int(dim)+1 {
		panic(fmt.Sprintf("Error: expected %d columns, got %d", dim+1, len(row)))
	}
	clusterIndex, err := utils.StringToUint64(row[0])
	if err != nil {
		panic("Error converting cluster index to uint64: " + err.Error())
	}
	query := make([]int8, dim)
	for i := 0; i < int(dim); i++ {
		u, err := strconv.ParseFloat(row[i+1], 64)
		query[i] = utils.QuantizeClamp(u, precBits)
		if err != nil {
			panic("Error converting query to int8: " + err.Error())
		}
	}
	return clusterIndex, query, false
}

type QueryPerf struct {
	clientHintQueryTime       time.Duration
	serverHintAnswerTime      time.Duration
	clientHintApplyTime       time.Duration
	clientQueryProcessingTime time.Duration
	serverComputeTime         time.Duration
	clientReconTime           time.Duration
	hintQuerySize             uint64
	hintAnsSize               uint64
	querySize                 uint64
	ansSize                   uint64
}

func writeResults(writer *csv.Writer, perfWriter *csv.Writer, scores *[]protocol.VectorScore, k int, perf *QueryPerf) {
	if len(*scores) == 0 {
		panic("Error: No scores to write")
	}
	numRes := k
	if numRes > len(*scores) {
		numRes = len(*scores)
	}
	line := make([]string, numRes*2)
	for i := 0; i < numRes; i++ {
		line[i*2] = fmt.Sprintf("%d", (*scores)[i].ClusterID)
		line[i*2+1] = fmt.Sprintf("%d", (*scores)[i].IDWithinCluster)
	}
	if err := writer.Write(line); err != nil {
		panic("Error writing to output file: " + err.Error())
	}
	writer.Flush()

	perfLine := []string{
		fmt.Sprintf("%d", perf.clientHintQueryTime.Milliseconds()),
		fmt.Sprintf("%d", perf.serverHintAnswerTime.Milliseconds()),
		fmt.Sprintf("%d", perf.clientHintApplyTime.Milliseconds()),
		fmt.Sprintf("%d", perf.clientQueryProcessingTime.Milliseconds()),
		fmt.Sprintf("%d", perf.serverComputeTime.Milliseconds()),
		fmt.Sprintf("%d", perf.clientReconTime.Milliseconds()),
		fmt.Sprintf("%d", perf.hintQuerySize),
		fmt.Sprintf("%d", perf.hintAnsSize),
		fmt.Sprintf("%d", perf.querySize),
		fmt.Sprintf("%d", perf.ansSize),
	}
	if err := perfWriter.Write(perfLine); err != nil {
		panic("Error writing to performance output file: " + err.Error())
	}
	perfWriter.Flush()
}

func main() {
	preamble := flag.String("preamble", "", "Preamble to use for the search")
	topK := flag.Int("topk", 10, "Number of top results to return")
	precBits := flag.Uint64("precBits", 5, "Number of bits to use for precision")
	clusterOnly := flag.Bool("clusterOnly", false, "Only return top k among vectors in the specified cluster")

	flag.Parse()
	argumentsValidation(*preamble, *topK)

	fmt.Printf("Preamble: %s\n", *preamble)
	fmt.Printf("Top K: %d\n", *topK)
	fmt.Printf("Cluster Only: %t\n", *clusterOnly)

	// start a timer
	serverPreProcessingStart := time.Now()
	metadata, clusters := database.ReadAllClusters(*preamble, *precBits)
	hintSz := uint64(900)

	server := new(protocol.Server)
	server.ProcessVectorsFromClusters(metadata, clusters, hintSz, *precBits)

	serverPreProcessingTime := time.Since(serverPreProcessingStart)

	fmt.Printf("Server database constrction time: %s\n", serverPreProcessingTime)

	client := new(protocol.Client)
	client.Setup(server.Hint)

	dir := filepath.Dir(*preamble)
	prefix := filepath.Base(*preamble)
	queryFile := utils.OpenFile(filepath.Join(dir, prefix+"_queries.csv"))
	defer queryFile.Close()

	reader := csv.NewReader(queryFile)

	outputFileSuffix := "_results.csv"
	if *clusterOnly {
		outputFileSuffix = "_results_cluster_only.csv"
	}
	outputFile, err := os.Create(filepath.Join(dir, prefix+outputFileSuffix))
	if err != nil {
		panic("Error creating output file: " + err.Error())
	}
	defer outputFile.Close()
	writer := csv.NewWriter(outputFile)
	defer writer.Flush()

	perfFileSuffix := "_perf.csv"
	if *clusterOnly {
		perfFileSuffix = "_perf_cluster_only.csv"
	}
	perfFile, err := os.Create(filepath.Join(dir, prefix+perfFileSuffix))
	if err != nil {
		panic("Error creating performance output file: " + err.Error())
	}
	defer perfFile.Close()
	perfWriter := csv.NewWriter(perfFile)
	defer perfWriter.Flush()

	// write the header for the perf csv
	perfHeader := []string{
		"clientHintQueryTime",
		"serverHintAnswerTime",
		"clientHintApplyTime",
		"clientQueryProcessingTime",
		"serverComputeTime",
		"clientReconTime",
		"hintQuerySize",
		"hintAnsSize",
		"querySize",
		"ansSize",
	}
	if err := perfWriter.Write(perfHeader); err != nil {
		panic("Error writing to performance output file: " + err.Error())
	}
	perfWriter.Flush()

	queryCount := 0
	for {
		clusterIndex, query, isEnd := readQueryLine(reader, metadata.Dim, *precBits)
		if isEnd {
			break
		}
		sortedScores, perf := runRound(client, server, query, clusterIndex, *clusterOnly)
		writeResults(writer, perfWriter, sortedScores, *topK, perf)
		queryCount++

		if queryCount%1000 == 0 {
			fmt.Printf("Processed %d queries\n", queryCount)
		}
	}
}

func runRound(c *protocol.Client, s *protocol.Server, query []int8, clusterIndex uint64, clusterOnly bool) (*[]protocol.VectorScore, *QueryPerf) {
	clientHintQuery := time.Now()
	ct := c.PreprocessQuery()
	clientHintQueryTime := time.Since(clientHintQuery)
	hintQuerySize := utils.MessageSizeBytes(*ct)

	serverHintAnswerStart := time.Now()
	offlineAns := s.HintAnswer(ct)
	serverHintAnswerTime := time.Since(serverHintAnswerStart)
	hintAnsSize := utils.MessageSizeBytes(*offlineAns)

	clientHintApplyStart := time.Now()
	c.ProcessHintApply(offlineAns)
	clientHintApplyTime := time.Since(clientHintApplyStart)

	clientQueryProcessingStart := time.Now()
	queryEmb := c.QueryEmbeddings(query, clusterIndex)
	clientQueryProcessingTime := time.Since(clientQueryProcessingStart)

	querySize := utils.MessageSizeBytes(*queryEmb)

	serverComputeStart := time.Now()
	ans := s.Answer(queryEmb)
	serverComputeTime := time.Since(serverComputeStart)
	ansSize := utils.MessageSizeBytes(*ans)

	var recon *[]protocol.VectorScore

	clientReconStart := time.Now()
	if clusterOnly {
		recon = c.ReconstructWithinCluster(ans, clusterIndex, c.DBInfo.P())
	} else {
		recon = c.ReconstructWithinBin(ans, clusterIndex, c.DBInfo.P())
	}
	clientReconTime := time.Since(clientReconStart)

	perf := &QueryPerf{
		clientHintQueryTime:       clientHintQueryTime,
		serverHintAnswerTime:      serverHintAnswerTime,
		clientHintApplyTime:       clientHintApplyTime,
		clientQueryProcessingTime: clientQueryProcessingTime,
		serverComputeTime:         serverComputeTime,
		clientReconTime:           clientReconTime,
		hintQuerySize:             hintQuerySize,
		hintAnsSize:               hintAnsSize,
		querySize:                 querySize,
		ansSize:                   ansSize,
	}

	return recon, perf
}
