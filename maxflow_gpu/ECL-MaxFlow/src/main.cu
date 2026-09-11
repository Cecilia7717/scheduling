/*
ECL-MaxFlow: This code computes the maximum flow of a directed or undirected graph.

Copyright (c) 2025, Avery VanAusdal and Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at
https://github.com/burtscher/ECL-MaxFlow.

Publication: This work is described in detail in the following paper.
Avery Vanausdal and Martin Burtscher. An Efficient Push-Relabel Implementation
for Max-Flow Computations on GPUs. Proceedings of the 44th IEEE International
Performance Computing and Communications Conference. November 2025.

Sponsor: This code is based upon work supported by the U.S. National Science
Foundation (NSF) under Award #1955367 and by an equipment donation from
NVIDIA Corporation.
*/

#include "maxflow.cu"
#include <fstream>


int main(int argc, char* argv[])
{
  printf("ECL-Maxflow\n");
  printf("Copyright 2025 Avery VanAusdal and Martin Burtscher\n");

  if (argc < 4) {
    fprintf(
        stderr,
        "USAGE: %s input_file_name source_node sink_node "
        "runs(default=%i) output_csv(optional)\n\n"
        "The optional 'runs' parameter runs the code multiple times "
        "and returns the median runtime and throughput.\n"
        "The optional 'output_csv' parameter specifies where the final "
        "edge flows are written.\n",
        argv[0],
        default_runs
    );
    exit(-1);
  }


  GPUTimer header_timer, rcsr_timer;
  header_timer.start();


  // ============================================================
  // Process command line
  // ============================================================

  ECLgraph g = readECLgraph(argv[1]);

  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);


  const int source = atoi(argv[2]);

  if ((source < 0) || (source >= g.nodes)) {
    fprintf(
        stderr,
        "ERROR: source_node must be between 0 and %d\n",
        g.nodes - 1
    );
    exit(-1);
  }

  printf("source: %d\n", source);


  const int sink = atoi(argv[3]);

  if ((sink < 0) || (sink >= g.nodes)) {
    fprintf(
        stderr,
        "ERROR: sink_node must be between 0 and %d\n",
        g.nodes - 1
    );
    exit(-1);
  }

  if (sink == source) {
    fprintf(
        stderr,
        "ERROR: sink_node and source_node cannot be the same node\n"
    );
    exit(-1);
  }

  printf("sink:   %d\n", sink);


  int runs = default_runs;

  if (argc >= 5) {
    const int runsInt = atoi(argv[4]);

    if (runsInt > 0) {
      runs = runsInt;
    }
  }

  printf("runs: %d\n", runs);


  // Optional output CSV filename.
  //
  // Example:
  //
  // ./maxflow graph.egr 0 16 1 pass1_flow.csv
  //
  // If omitted:
  //
  // ./maxflow graph.egr 0 16
  //
  // the default file is flow_output.csv.

  const char* csv_filename =
      (argc >= 6)
          ? argv[5]
          : "flow_output.csv";

  printf("flow output: %s\n", csv_filename);


  // ============================================================
  // GPU information
  // ============================================================

  GPUinfo(0);


  // ============================================================
  // Allocate host/device graph structures
  // ============================================================

  int* const flow = new int[g.edges];

  // GPUmaxflow writes the resulting flow back into this host array.
  int* const capacity = new int[g.edges];


  ECLgraph d_g = g;

  if (
      cudaSuccess !=
      cudaMalloc(
          (void**)&d_g.nindex,
          (g.nodes + 1) * sizeof(int)
      )
  ) {
    fprintf(
        stderr,
        "ERROR: could not allocate nindex\n"
    );
    exit(-1);
  }

  if (
      cudaSuccess !=
      cudaMalloc(
          (void**)&d_g.nlist,
          g.edges * sizeof(int)
      )
  ) {
    fprintf(
        stderr,
        "ERROR: could not allocate nlist\n"
    );
    exit(-1);
  }

  if (
      cudaSuccess !=
      cudaMemcpy(
          d_g.nindex,
          g.nindex,
          (g.nodes + 1) * sizeof(int),
          cudaMemcpyHostToDevice
      )
  ) {
    fprintf(
        stderr,
        "ERROR: copying of index to device failed\n"
    );
    exit(-1);
  }

  if (
      cudaSuccess !=
      cudaMemcpy(
          d_g.nlist,
          g.nlist,
          g.edges * sizeof(int),
          cudaMemcpyHostToDevice
      )
  ) {
    fprintf(
        stderr,
        "ERROR: copying of nlist to device failed\n"
    );
    exit(-1);
  }


  // ============================================================
  // Create reverse CSR structure for backwards traversal
  // ============================================================

  int* const rnindex = new int[g.nodes + 1];
  int* const rnlist = new int[g.edges];

  // Maps reverse edge index to the corresponding original edge:
  //
  // retoe[reverse_edge] = original_edge
  int* const retoe = new int[g.edges];


  rcsr_timer.start();


  std::vector<std::pair<int, int>>* const incoming =
      new std::vector<std::pair<int, int>>[g.nodes];


  for (int v = 0; v < g.nodes; v++) {

    for (
        int e = g.nindex[v];
        e < g.nindex[v + 1];
        e++
    ) {

      const int nbor = g.nlist[e];

      incoming[nbor].push_back(
          std::make_pair(e, v)
      );
    }
  }


  int reidx = 0;

  rnindex[0] = 0;


  for (int v = 0; v < g.nodes; v++) {

    for (
        std::pair<int, int> p :
        incoming[v]
    ) {

      const int e = p.first;
      const int src = p.second;

      rnlist[reidx] = src;
      retoe[reidx] = e;

      ++reidx;
    }

    rnindex[v + 1] = reidx;
  }


  delete[] incoming;


  const double rcsr_time =
      rcsr_timer.stop();


  // ============================================================
  // Assign capacities
  // ============================================================

  if (g.eweight == NULL) {

    // No capacities in the ECL graph.
    // Fall back to ECL's original random-capacity behavior.

    srand(source);

    for (
        int e = 0;
        e < g.edges;
        e++
    ) {

      capacity[e] =
          rand() % g.nodes;
    }

  } else {

    // DIMACS converter stores capacities as edge weights.

    for (
        int e = 0;
        e < g.edges;
        e++
    ) {

      capacity[e] =
          abs(g.eweight[e]);
    }
  }


  // ============================================================
  // Print source/sink information
  // ============================================================

  long sink_cap = 0;


  for (
      int re = rnindex[sink];
      re < rnindex[sink + 1];
      re++
  ) {

    const int e = retoe[re];

    sink_cap +=
        capacity[e];
  }


  printf(
      "source info: %i in-edges, %i out-edges\n",
      rnindex[source + 1] - rnindex[source],
      g.nindex[source + 1] - g.nindex[source]
  );


  printf(
      "sink max capacity: %li across %i in-edges\n",
      sink_cap,
      rnindex[sink + 1] - rnindex[sink]
  );


  // ============================================================
  // Copy reverse CSR to GPU
  // ============================================================

  int* d_rnindex;


  if (
      cudaSuccess !=
      cudaMalloc(
          (void**)&d_rnindex,
          (g.nodes + 1) * sizeof(int)
      )
  ) {
    fprintf(
        stderr,
        "ERROR: could not allocate reverse nindex\n"
    );
    exit(-1);
  }


  if (
      cudaSuccess !=
      cudaMemcpy(
          d_rnindex,
          rnindex,
          (g.nodes + 1) * sizeof(int),
          cudaMemcpyHostToDevice
      )
  ) {
    fprintf(
        stderr,
        "ERROR: copying reverse index to device failed\n"
    );
    exit(-1);
  }


  delete[] rnindex;


  int* d_rnlist;


  if (
      cudaSuccess !=
      cudaMalloc(
          (void**)&d_rnlist,
          g.edges * sizeof(int)
      )
  ) {
    fprintf(
        stderr,
        "ERROR: could not allocate reverse nlist\n"
    );
    exit(-1);
  }


  if (
      cudaSuccess !=
      cudaMemcpy(
          d_rnlist,
          rnlist,
          g.edges * sizeof(int),
          cudaMemcpyHostToDevice
      )
  ) {
    fprintf(
        stderr,
        "ERROR: copying reverse nlist to device failed\n"
    );
    exit(-1);
  }


  delete[] rnlist;


  int* d_retoe;


  if (
      cudaSuccess !=
      cudaMalloc(
          (void**)&d_retoe,
          g.edges * sizeof(int)
      )
  ) {
    fprintf(
        stderr,
        "ERROR: could not allocate retoe\n"
    );
    exit(-1);
  }


  if (
      cudaSuccess !=
      cudaMemcpy(
          d_retoe,
          retoe,
          g.edges * sizeof(int),
          cudaMemcpyHostToDevice
      )
  ) {
    fprintf(
        stderr,
        "ERROR: copying retoe to device failed\n"
    );
    exit(-1);
  }


  delete[] retoe;


  printf(
      "header total init time: %.6fs "
      "(including building rcsr: %.6fs)\n",
      header_timer.stop(),
      rcsr_time
  );


  // ============================================================
  // Run GPU max flow
  // ============================================================

  double* const runtimes =
      new double[runs];


  for (
      int i = 0;
      i < runs;
      i++
  ) {

    runtimes[i] =
        GPUmaxflow(
            g,
            d_g,
            source,
            sink,
            flow,
            capacity,
            d_rnindex,
            d_rnlist,
            d_retoe
        );

    fflush(NULL);
  }


  // ============================================================
  // Write FINAL flow from the last run to CSV
  // ============================================================
  //
  // Note:
  //   GPUmaxflow resets and computes a fresh flow for each run.
  //   Therefore flow[] after this loop corresponds to the LAST run.
  //
  // For scheduling/debugging we normally call:
  //
  //     runs = 1
  //
  // so this is exactly the flow used by the scheduling pipeline.
  // ============================================================

  std::ofstream csv(csv_filename);


  if (!csv.is_open()) {

    fprintf(
        stderr,
        "ERROR: could not open %s for writing\n",
        csv_filename
    );

    exit(-1);
  }


  csv <<
      "from,to,flow,capacity\n";


  for (
      int u = 0;
      u < g.nodes;
      u++
  ) {

    for (
        int e = g.nindex[u];
        e < g.nindex[u + 1];
        e++
    ) {

      const int v =
          g.nlist[e];


      csv
          << u
          << ","
          << v
          << ","
          << flow[e]
          << ","
          << capacity[e]
          << "\n";
    }
  }


  csv.close();


  if (!csv) {

    fprintf(
        stderr,
        "ERROR: failed while writing CSV file %s\n",
        csv_filename
    );

    exit(-1);
  }


  printf(
      "Final edge flows written to: %s\n",
      csv_filename
  );


  // ============================================================
  // Print final edge flow table
  // ============================================================

  printf("\n");

  printf(
      "============================================================\n"
  );

  printf(
      "FINAL EDGE FLOWS\n"
  );

  printf(
      "============================================================\n"
  );

  printf(
      "%-8s %-8s %-10s %-10s\n",
      "FROM",
      "TO",
      "FLOW",
      "CAPACITY"
  );

  printf(
      "------------------------------------------------------------\n"
  );


  for (
      int u = 0;
      u < g.nodes;
      u++
  ) {

    for (
        int e = g.nindex[u];
        e < g.nindex[u + 1];
        e++
    ) {

      const int v =
          g.nlist[e];


      printf(
          "%-8d %-8d %-10d %-10d\n",
          u,
          v,
          flow[e],
          capacity[e]
      );
    }
  }


  printf(
      "============================================================\n\n"
  );


  // ============================================================
  // Runtime summary
  // ============================================================

  const double med =
      median(
          runtimes,
          runs
      );


  printf(
      "median runtime: %.6fs\n",
      med
  );


  printf(
      "Throughput: %.6f gigaedges/s\n",
      0.000000001 *
      g.edges /
      med
  );


  // ============================================================
  // Cleanup
  // ============================================================

  cudaFree(
      d_g.nindex
  );

  cudaFree(
      d_g.nlist
  );

  cudaFree(
      d_rnindex
  );

  cudaFree(
      d_rnlist
  );

  cudaFree(
      d_retoe
  );


  delete[] flow;
  delete[] capacity;
  delete[] runtimes;


  return 0;
}