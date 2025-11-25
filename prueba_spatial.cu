#include <mpi.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#include <cstdio>
#include <cfloat>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, err); \
        } \
    } while (0)

using LaneCell = int;
constexpr LaneCell EMPTY_SLOT = -1;
#define MAX_ROUTE_SIZE 10

struct Edge {
    int id;
    int u, v;
    float length_meters;
    int num_lanes;
    float speed_mph;
    size_t lane_map_start_index;
};

struct Node {
    long osmid;
    double x, y;
    std::string ref;
    std::string highway;
    int id;
};

struct Vehicle {
    int id;
    bool active;
    int route[MAX_ROUTE_SIZE];
    int route_index;
    int current_edge_id;
    int current_lane;
    int position_on_edge;
    int speed_mps;
    int desired_speed_mps;

    __device__ void update(const Edge* d_edges,
                           LaneCell* d_cur,
                           LaneCell* d_next,
                           const int* d_edge_owner_map)
    {
        if (!active) return;

        const Edge& cur = d_edges[current_edge_id];

        int len = (int)cur.length_meters;
        size_t idx = cur.lane_map_start_index +
                     (size_t)current_lane * len +
                     position_on_edge;

        if (idx >= cur.lane_map_start_index + (size_t)(cur.num_lanes * len)) {
             active = false;
             return;
        }

        bool obstacle = false;
        int look_ahead = speed_mps + 5;
        for (int d = 1; d <= look_ahead; ++d) {
            int p = position_on_edge + d;
            if (p >= len) break;
            size_t idx2 = cur.lane_map_start_index +
                          (size_t)current_lane * len +
                          p;
            if (idx2 >= cur.lane_map_start_index + (size_t)(cur.num_lanes * len)) break;
            if (d_cur[idx2] != EMPTY_SLOT) {
                obstacle = true;
                break;
            }
        }

        if (obstacle) {
            speed_mps = 0;
        } else if (speed_mps < desired_speed_mps) {
            speed_mps++;
        }

        int next_pos  = position_on_edge + speed_mps;
        int next_edge = current_edge_id;
        int next_lane = current_lane;

        if (next_pos >= len) {
            route_index++;
            if (route_index >= MAX_ROUTE_SIZE || route[route_index] == -1) {
                active = false;
                return;
            }
            next_edge = route[route_index];
            next_pos  = 0;
        }

        const Edge& nxt = d_edges[next_edge];
        int next_len = (int)nxt.length_meters;

        size_t idx_next = nxt.lane_map_start_index +
                          (size_t)next_lane * next_len +
                          next_pos;
        
        if (idx_next >= nxt.lane_map_start_index + (size_t)(nxt.num_lanes * next_len)) {
            active = false;
            return;
        }

        LaneCell expected = EMPTY_SLOT;
        LaneCell mine     = id;

        if (atomicCAS(&d_next[idx_next], expected, mine) == expected) {
            atomicExch(&d_next[idx], EMPTY_SLOT);

            current_edge_id   = next_edge;
            position_on_edge  = next_pos;
        } else {
            speed_mps = 0;
            atomicCAS(&d_next[idx], EMPTY_SLOT, mine);
        }
    }
};

__global__ void update_kernel(Vehicle* d_veh,
                              int n,
                              const Edge* d_edges,
                              LaneCell* d_cur,
                              LaneCell* d_next,
                              const int* d_edge_owner_map)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_veh[i].update(d_edges, d_cur, d_next, d_edge_owner_map);
    }
}

std::unordered_map<long, int> osmid_to_index;

std::vector<Node> load_nodes_csv(const std::string& path) {
    std::vector<Node> nodes;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error abriendo " << path << "\n";
        return nodes;
    }

    std::string line;
    getline(f, line);
    while (getline(f, line)) {
        std::stringstream ss(line);
        std::string cell;
        Node n;
        std::getline(ss, cell, ','); n.osmid = std::stol(cell);
        std::getline(ss, cell, ','); n.x     = std::stod(cell);
        std::getline(ss, cell, ','); n.y     = std::stod(cell);
        std::getline(ss, cell, ','); n.ref   = cell;
        std::getline(ss, cell, ','); n.highway = cell;
        std::getline(ss, cell, ','); n.id    = std::stoi(cell);

        nodes.push_back(n);
        osmid_to_index[n.osmid] = n.id;
    }
    return nodes;
}

std::vector<Edge> load_edges_csv(const std::string& path) {
    std::vector<Edge> edges;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error abriendo " << path << "\n";
        return edges;
    }

    std::string line;
    getline(f, line);
    size_t offset = 0;
    while (getline(f, line)) {
        std::stringstream ss(line);
        std::string cell;
        Edge e;

        std::getline(ss, cell, ','); e.id            = std::stoi(cell);
        std::getline(ss, cell, ',');
        std::getline(ss, cell, ',');
        std::getline(ss, cell, ','); e.length_meters = std::stof(cell);
        std::getline(ss, cell, ','); e.num_lanes    = std::stoi(cell);
        std::getline(ss, cell, ','); e.speed_mph    = std::stof(cell);
        std::getline(ss, cell, ','); e.u            = std::stoi(cell);
        std::getline(ss, cell, ','); e.v            = std::stoi(cell);

        e.lane_map_start_index = offset;
        offset += (size_t)e.length_meters * e.num_lanes;
        edges.push_back(e);
    }
    return edges;
}

std::vector<Vehicle> load_od_csv(const std::string& path,
                                 const std::vector<Edge>& edges)
{
    std::vector<Vehicle> veh;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error abriendo " << path << "\n";
        return veh;
    }

    std::string line;
    getline(f, line);
    int id = 0;
    while (getline(f, line)) {
        std::stringstream ss(line);
        std::string cell;
        Vehicle v;

        std::getline(ss, cell, ',');
        std::getline(ss, cell, ',');
        std::getline(ss, cell, ','); long orig_osmid = std::stol(cell);
        std::getline(ss, cell, ','); long dest_osmid = std::stol(cell);

        auto it_o = osmid_to_index.find(orig_osmid);
        auto it_d = osmid_to_index.find(dest_osmid);
        if (it_o == osmid_to_index.end() || it_d == osmid_to_index.end()) {
            continue;
        }

        int orig_idx = it_o->second;
        int dest_idx = it_d->second;

        v.id = ++id;
        v.active = true;
        
        int start_edge_id = -1;
        for (const auto& e : edges) {
            if (e.u == orig_idx) {
                start_edge_id = e.id;
                break;
            }
        }

        if (start_edge_id == -1) {
            if (!edges.empty()) {
                start_edge_id = orig_idx % (int)edges.size();
                if (start_edge_id < 0) start_edge_id = 0;
            } else {
                continue;
            }
        }
        
        int end_edge_id = -1;
        for (const auto& e : edges) {
            if (e.v == dest_idx || e.u == dest_idx) {
                end_edge_id = e.id;
                break;
            }
        }
        if (end_edge_id == -1) {
            if (!edges.empty()) {
                end_edge_id = dest_idx % (int)edges.size();
                if (end_edge_id < 0) end_edge_id = 0;
            } else {
                continue;
            }
        }

        v.route[0] = start_edge_id;
        v.route[1] = end_edge_id;
        for (int i = 2; i < MAX_ROUTE_SIZE; ++i) v.route[i] = -1;

        v.route_index     = 0;
        v.current_edge_id = v.route[0];
        v.current_lane    = 0;
        v.position_on_edge = 0;
        v.speed_mps       = 0;
        v.desired_speed_mps = 10;

        veh.push_back(v);
    }
    return veh;
}

int exchange_and_update_gpu(thrust::device_vector<Vehicle>& d_veh,
                            std::vector<Vehicle>& h_veh_buffer,
                            int my_rank, int world_size,
                            const std::vector<int>& h_edge_owner_map)
{
    if (world_size == 1) {
        return 0;
    }

    h_veh_buffer.resize(d_veh.size());
    thrust::copy(d_veh.begin(), d_veh.end(), h_veh_buffer.begin());

    std::vector<Vehicle> staying_vehs;
    std::vector<Vehicle> outgoing_vehs;
    staying_vehs.reserve(h_veh_buffer.size());
    outgoing_vehs.reserve(h_veh_buffer.size() / 10);

    for (const auto& v : h_veh_buffer) {
        if (!v.active) continue;
        int owner_rank = h_edge_owner_map[v.current_edge_id];
        if (owner_rank == my_rank) staying_vehs.push_back(v);
        else                       outgoing_vehs.push_back(v);
    }

    int other_rank = 1 - my_rank;
    std::vector<Vehicle> incoming_vehs;

    int send_count = (int)outgoing_vehs.size();
    int recv_count = 0;

    MPI_Sendrecv(&send_count, 1, MPI_INT, other_rank, 0,
                 &recv_count, 1, MPI_INT, other_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (recv_count > 0) {
        incoming_vehs.resize(recv_count);
    }

    int send_bytes = send_count * (int)sizeof(Vehicle);
    int recv_bytes = recv_count * (int)sizeof(Vehicle);

    void* send_buf = (send_count > 0) ? (void*)outgoing_vehs.data() : nullptr;
    void* recv_buf = (recv_count > 0) ? (void*)incoming_vehs.data() : nullptr;

    MPI_Sendrecv(send_buf, send_bytes, MPI_BYTE, other_rank, 1,
                 recv_buf, recv_bytes, MPI_BYTE, other_rank, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    h_veh_buffer.clear();
    h_veh_buffer.reserve(staying_vehs.size() + incoming_vehs.size());
    h_veh_buffer.insert(h_veh_buffer.end(),
                        staying_vehs.begin(), staying_vehs.end());
    h_veh_buffer.insert(h_veh_buffer.end(),
                        incoming_vehs.begin(), incoming_vehs.end());

    d_veh.resize(h_veh_buffer.size());
    thrust::copy(h_veh_buffer.begin(), h_veh_buffer.end(), d_veh.begin());

    return send_count;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 1 || size > 2) {
        if (rank == 0)
            std::cerr << "Error: esta prueba requiere 1 o 2 procesos MPI.\n";
        MPI_Finalize();
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    if (rank == 0)
        std::cout << "=== Simulación MultiGPU MPI (Particionamiento Espacial por Franjas) ===\n"
                  << "Nodos: " << size << "\n";

    std::vector<Edge>     edges;
    std::vector<Node>     nodes;
    std::vector<Vehicle>  vehicles;
    std::vector<int>      h_edge_owner_map;

    std::string data_dir = "/tmp/dataset/";

    if (rank == 0) {
        nodes    = load_nodes_csv(data_dir + "nodes.csv");
        edges    = load_edges_csv(data_dir + "edges.csv");
        vehicles = load_od_csv(data_dir + "od_demand.csv", edges);

        std::cout << "Datos cargados en Rank 0. Edges: " << edges.size()
                  << ", Nodes: " << nodes.size()
                  << ", Vehículos: " << vehicles.size() << "\n";
    }

    int num_edges = (int)edges.size();
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) edges.resize(num_edges);

    const int CHUNK_SIZE_BYTES = 1024 * 1024;
    int total_bytes_edges = num_edges * (int)sizeof(Edge);
    int num_chunks = (total_bytes_edges + CHUNK_SIZE_BYTES - 1) / CHUNK_SIZE_BYTES;

    for (int i = 0; i < num_chunks; ++i) {
        int offset_bytes = i * CHUNK_SIZE_BYTES;
        int current_chunk_bytes = std::min(CHUNK_SIZE_BYTES, total_bytes_edges - offset_bytes);
        if (current_chunk_bytes <= 0) break;
        MPI_Bcast(reinterpret_cast<char*>(edges.data()) + offset_bytes,
                  current_chunk_bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
        std::cout << "DEBUG R" << rank << ": Edges difundidos. Num_edges: "
                  << num_edges << "\n";

    if (rank == 0 && size > 1) {
        h_edge_owner_map.resize(num_edges);

        double min_x = DBL_MAX;
        double max_x = -DBL_MAX;
        for (const auto& n : nodes) {
            if (n.x < min_x) min_x = n.x;
            if (n.x > max_x) max_x = n.x;
        }

        double split_x = (min_x + max_x) / 2.0;

        std::cout << "DEBUG R0: Particionamiento Espacial. Min X: " << min_x << ", Max X: " << max_x << ", Split X: " << split_x << "\n";

        for (const auto& e : edges) {
            if (nodes[e.u].x < split_x) {
                h_edge_owner_map[e.id] = 0;
            } else {
                h_edge_owner_map[e.id] = 1;
            }
        }
        std::cout << "DEBUG R0: Mapeo de propietarios de aristas calculado (Espacial).\n";
    } else if (rank == 0 && size == 1) {
        h_edge_owner_map.assign(num_edges, 0);
    }
    
    h_edge_owner_map.resize(num_edges);
    MPI_Bcast(h_edge_owner_map.data(), num_edges, MPI_INT, 0, MPI_COMM_WORLD);

    int total_vehicles = (int)vehicles.size();
    MPI_Bcast(&total_vehicles, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
        std::cout << "DEBUG R" << rank << ": Total vehicles bcast: "
                  << total_vehicles << "\n";

    if (total_vehicles == 0) {
        if (rank == 0) std::cout << "No hay vehículos. Terminando.\n";
        MPI_Finalize();
        return 0;
    }

    std::vector<int> sendcounts_bytes(size);
    std::vector<int> displs_bytes(size);
    int offset_bytes = 0;

    for (int i = 0; i < size; ++i) {
        int vehs = total_vehicles / size;
        if (i < (total_vehicles % size)) vehs++;

        int bytes = vehs * (int)sizeof(Vehicle);
        sendcounts_bytes[i] = bytes;
        displs_bytes[i]     = offset_bytes;
        offset_bytes       += bytes;
    }

    int local_bytes = sendcounts_bytes[rank];
    int local_veh_count = local_bytes / (int)sizeof(Vehicle);

    std::vector<Vehicle> local_veh(local_veh_count);

    if (rank == 0) {
        std::cout << "DEBUG R0: Scatterv vehs. sendcounts_bytes: ";
        for (auto c : sendcounts_bytes) std::cout << c << " ";
        std::cout << "\nDEBUG R0: displs_bytes: ";
        for (auto d : displs_bytes) std::cout << d << " ";
        std::cout << "\n";
    }

    MPI_Scatterv(
        (rank == 0 && !vehicles.empty()) ? (void*)vehicles.data() : nullptr,
        sendcounts_bytes.data(),
        displs_bytes.data(),
        MPI_BYTE,
        local_veh.data(),
        local_bytes,
        MPI_BYTE,
        0,
        MPI_COMM_WORLD
    );

    std::cout << "DEBUG R" << rank << ": Vehículos locales recibidos: " << local_veh.size() << "\n";

    thrust::device_vector<Edge>    d_edges = edges;
    thrust::device_vector<Vehicle> d_veh   = local_veh;
    thrust::device_vector<int>     d_edge_owner_map = h_edge_owner_map;

    size_t total_lane_cells = 0;
    for (auto& e : edges)
        total_lane_cells += (size_t)((int)e.length_meters) * e.num_lanes;

    if (rank == 0)
        std::cout << "DEBUG R" << rank << ": Total Lane Cells: " << total_lane_cells << "\n";

    LaneCell *d_cur = nullptr, *d_next = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cur,  total_lane_cells * sizeof(LaneCell)));
    CUDA_CHECK(cudaMalloc(&d_next, total_lane_cells * sizeof(LaneCell)));
    CUDA_CHECK(cudaMemset(d_cur,  0xFF, total_lane_cells * sizeof(LaneCell)));
    CUDA_CHECK(cudaMemset(d_next, 0xFF, total_lane_cells * sizeof(LaneCell)));

    int steps = (argc > 1) ? std::atoi(argv[1]) : 100;

    std::vector<Vehicle> h_veh_buffer;
    long long total_migrations_local = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::cout << "--- Inicio de la simulación ---\n";

    auto start_t = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < steps; ++t) {
        int n_local = (int)d_veh.size();

        if (n_local > 0) {
            int threadsPerBlock = 256;
            int blocks = (n_local + threadsPerBlock - 1) / threadsPerBlock;

            update_kernel<<<blocks, threadsPerBlock>>>(
                thrust::raw_pointer_cast(d_veh.data()),
                n_local,
                thrust::raw_pointer_cast(d_edges.data()),
                d_cur,
                d_next,
                thrust::raw_pointer_cast(d_edge_owner_map.data())
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        int migrated = exchange_and_update_gpu(d_veh, h_veh_buffer, rank, size, h_edge_owner_map);
        total_migrations_local += migrated;

        std::swap(d_cur, d_next);
        CUDA_CHECK(cudaMemset(d_next, 0xFF, total_lane_cells * sizeof(LaneCell)));

        MPI_Barrier(MPI_COMM_WORLD);

        if (t % 10 == 0 && rank == 0) {
            std::cout << "Step " << t << " completado. Migraciones en este paso (R0): " << migrated << "\n";
        }
    }

    auto end_t = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_t - start_t).count();

    long long global_migrations = 0;
    MPI_Reduce(&total_migrations_local, &global_migrations,
               1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "=== Fin de Simulación ===\n";
        std::cout << "Tiempo total: " << elapsed << " s\n";
        std::cout << "Total Migraciones en el Clúster: " << global_migrations << "\n";
    }

    CUDA_CHECK(cudaFree(d_cur));
    CUDA_CHECK(cudaFree(d_next));
    MPI_Finalize();
    return 0;
}
