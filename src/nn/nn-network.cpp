#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h> // For inet_addr and other functions
#include <windows.h>  // For SSIZE_T
typedef SSIZE_T ssize_t;
#else
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>  // for getaddrinfo
#endif
#include "nn-network.hpp"
#include "nn-core.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <fcntl.h>

#define SOCKET_LAST_ERRCODE errno
#define SOCKET_LAST_ERROR strerror(errno)

#define ACK 23571114
#define MAX_CHUNK_SIZE 65536

static inline bool isEagainError() {
    #ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK;
    #else
    return SOCKET_LAST_ERRCODE == EAGAIN;
    #endif
}

static NnUint getSplitTotal(const NnDimSplit& split, NnUint nNodes) {
    if (!split.lengths) return 0;
    NnUint sum = 0;
    for(NnUint i=0; i<nNodes; ++i) sum += split.lengths[i];
    return sum;
}

static NnUint getGroupRootIndex(const NnStageConfig* stage) {
    if (stage != nullptr) {
        // 如果是在某个 Stage 内部同步，Root 是该 Stage 的第一个节点
        return stage->rootNodeIndex; 
    }
    // 如果是全局同步，Root 是全局 Node 0
    return 0;
}

static inline void setNonBlocking(int socket, bool enabled) {
#ifdef _WIN32
    u_long mode = enabled ? 1 : 0;
    if (ioctlsocket(socket, FIONBIO, &mode) != 0) {
        throw std::runtime_error("Error setting socket to non-blocking");
    }
#else
    int flags = fcntl(socket, F_GETFL, 0);
    if (enabled) {
        flags |= O_NONBLOCK;
    } else {
        flags = flags & (~O_NONBLOCK);
    }
    if (fcntl(socket, F_SETFL, flags) < 0)
        throw std::runtime_error("Error setting socket to non-blocking");
#endif
}

static inline void setNoDelay(int socket) {
    int flag = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int)) < 0)
        throw std::runtime_error("Error setting socket to no-delay");
}

static inline void setQuickAck(int socket) {
#ifndef _WIN32
#ifdef TCP_QUICKACK
    int value = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_QUICKACK, (char*)&value, sizeof(int)) < 0)
        throw std::runtime_error("Error setting quick ack");
#endif
#endif
}

void setReuseAddr(int socket) {
    int opt = 1;
    #ifdef _WIN32
    int iresult = setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
    if (iresult == SOCKET_ERROR) {
        closesocket(socket);
        throw std::runtime_error("setsockopt failed: " + std::to_string(WSAGetLastError()));
    }
    #else
    if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        close(socket);
        throw std::runtime_error("setsockopt failed: " + std::string(strerror(errno)));
    }
    #endif
}

static NnUint getUnevenSliceSize(const NnUnevenPartitionPlan *plan, NnUint totalSize, NnUint nodeIndex) {
    if (!plan) return totalSize / plan->nNodes; // Fallback

    // 尝试匹配 Vocab
    NnUint vocabTotal = getSplitTotal(plan->vocabSplit, plan->nNodes);
    if (vocabTotal > 0 && totalSize % vocabTotal == 0) {
        return plan->vocabSplit.lengths[nodeIndex] * (totalSize / vocabTotal);
    }
    // 尝试匹配 FFN
    NnUint ffnTotal = getSplitTotal(plan->ffnSplit, plan->nNodes);
    if (ffnTotal > 0 && totalSize % ffnTotal == 0) {
        return plan->ffnSplit.lengths[nodeIndex] * (totalSize / ffnTotal);
    }
    // 尝试匹配 Heads
    NnUint headTotal = getSplitTotal(plan->headSplit, plan->nNodes);
    if (headTotal > 0 && totalSize % headTotal == 0) {
        return plan->headSplit.lengths[nodeIndex] * (totalSize / headTotal);
    }
    
    // 默认均匀
    return totalSize / plan->nNodes;
}

static NnUint getUnevenSliceOffset(const NnUnevenPartitionPlan *plan, NnUint totalSize, NnUint nodeIndex) {
    if (!plan) return (totalSize / plan->nNodes) * nodeIndex;

    // (同上逻辑，使用 starts 而不是 lengths)
    NnUint vocabTotal = getSplitTotal(plan->vocabSplit, plan->nNodes);
    if (vocabTotal > 0 && totalSize % vocabTotal == 0) {
        return plan->vocabSplit.starts[nodeIndex] * (totalSize / vocabTotal);
    }
    // ... FFN, Heads ...
    
    return (totalSize / plan->nNodes) * nodeIndex;
}

static void fillUnevenSlices(const NnUnevenPartitionPlan *plan, NnUint nNodes, NnSize totalBytes, 
                             std::vector<NnSize>& offsets, std::vector<NnSize>& sizes) {
    bool matchFound = false;

    if (plan && plan->nNodes == nNodes) {
        // Helper lambda to check if a specific split configuration matches the current buffer size
        auto tryMatch = [&](const NnDimSplit& split) -> bool {
            NnUint totalUnits = getSplitTotal(split, nNodes);
            // Check if totalBytes is exactly divisible by the total units (e.g. total heads)
            // This is a heuristic: if we have 152k bytes and 152k vocab, it matches.
            if (totalUnits > 0 && totalBytes % totalUnits == 0) {
                NnSize bytesPerUnit = totalBytes / totalUnits;
                
                NnSize currentOffset = 0;
                for (NnUint i = 0; i < nNodes; ++i) {
                    NnSize len = split.lengths[i] * bytesPerUnit;
                    offsets[i] = currentOffset; // Recalculate offset to be safe
                    sizes[i] = len;
                    currentOffset += len;
                }
                return true;
            }
            return false;
        };

        // Priority order for matching:
        // 1. Vocab (Logits) - Largest, usually most critical for the "degeneration" bug
        if (!matchFound) matchFound = tryMatch(plan->vocabSplit);
        // 2. FFN - Intermediate layers
        if (!matchFound) matchFound = tryMatch(plan->ffnSplit);
        // 3. Dim - General dimension splits
        if (!matchFound) matchFound = tryMatch(plan->dimSplit);
        // 4. Heads - Attention Q
        if (!matchFound) matchFound = tryMatch(plan->headSplit);
        // 5. KV Heads - Attention K/V
        if (!matchFound) matchFound = tryMatch(plan->kvHeadSplit);
    }

    // Fallback: Uniform partitioning
    if (!matchFound) {
        NnSize avgBytes = totalBytes / nNodes;
        for (NnUint i = 0; i < nNodes; ++i) {
            sizes[i] = avgBytes;
            offsets[i] = i * avgBytes;
        }
        // Fix rounding error for the last node
        offsets[nNodes - 1] = (nNodes - 1) * avgBytes;
        sizes[nNodes - 1] = totalBytes - offsets[nNodes - 1];
    }
}

void writeSocket(int socket, const void *data, NnSize size) {
    while (size > 0) {
        ssize_t s = send(socket, (const char*)data, size, 0);
        if (s < 0) {
            if (isEagainError()) {
                continue;
            }
            throw NnTransferSocketException(0, "Error writing to socket");
        } else if (s == 0) {
            throw NnTransferSocketException(0, "Socket closed");
        }
        size -= s;
        data = (const char*)data + s;
    }
}

static inline bool tryReadSocket(int socket, void *data, NnSize size, unsigned long maxAttempts) {
    // maxAttempts = 0 means infinite attempts
    NnSize s = size;
    while (s > 0) {
        ssize_t r = recv(socket, (char*)data, s, 0);
        if (r < 0) {
            if (isEagainError()) {
                if (s == size && maxAttempts > 0) {
                    maxAttempts--;
                    if (maxAttempts == 0) {
                        return false;
                    }
                }
                continue;
            }
            throw NnTransferSocketException(0, "Error reading from socket");
        } else if (r == 0) {
            throw NnTransferSocketException(0, "Socket closed");
        }
        data = (char*)data + r;
        s -= r;
    }
    return true;
}

void readSocket(int socket, void *data, NnSize size) {
    if (!tryReadSocket(socket, data, size, 0)) {
        throw std::runtime_error("Error reading from socket");
    }
}

static void readAckPacket(int socket) {
    NnUint packet;
    readSocket(socket, &packet, sizeof(packet));
    if (packet != ACK)
        throw std::runtime_error("Invalid ack packet");
}

static void writeAckPacket(int socket) {
    NnUint packet = ACK;
    writeSocket(socket, &packet, sizeof(packet));
}

static inline int connectSocket(char *host, int port) {
    struct addrinfo hints;
    struct addrinfo *addr = NULL;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    char portStr[11];
    snprintf(portStr, sizeof(portStr), "%d", port);

    int addrinfoError = getaddrinfo(host, portStr, &hints, &addr);
    if (addrinfoError != 0 || addr == NULL) {
        printf("Cannot resolve target %s (%s)\n", host, gai_strerror(addrinfoError));
        throw NnConnectionSocketException("Cannot resolve address");
    }

    int sock = ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (sock < 0)
        throw std::runtime_error("Cannot create socket");

    int connectResult = ::connect(sock, addr->ai_addr, addr->ai_addrlen);
    if (connectResult != 0) {
        printf("Cannot connect to %s:%d (%s)\n", host, port, SOCKET_LAST_ERROR);
        throw NnConnectionSocketException("Cannot connect");
    }

    setNoDelay(sock);
    setQuickAck(sock);
    return sock;
}

int createServerSocket(int port) {
    const char *host = "0.0.0.0";
    struct sockaddr_in serverAddr;

    int serverSocket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (serverSocket < 0)
        throw std::runtime_error("Cannot create socket");
    setReuseAddr(serverSocket);

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = inet_addr(host);

    int bindResult;
    #ifdef _WIN32
    bindResult = bind(serverSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr));
    if (bindResult == SOCKET_ERROR) {
        int error = WSAGetLastError();
        closesocket(serverSocket);
        throw std::runtime_error("Cannot bind port: " + std::to_string(error));
    }
    #else
    bindResult = bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    if (bindResult < 0) {
        close(serverSocket);
        throw std::runtime_error("Cannot bind port: " + std::string(strerror(errno)));
    }
    #endif

    int listenResult = listen(serverSocket, SOMAXCONN);
    if (listenResult != 0) {
        #ifdef _WIN32
        closesocket(serverSocket);
        throw std::runtime_error("Cannot listen on port: " + std::to_string(WSAGetLastError()));
        #else
        close(serverSocket);
        throw std::runtime_error("Cannot listen on port: " + std::string(strerror(errno)));
        #endif
    }

    printf("Listening on %s:%d...\n", host, port);

    setNoDelay(serverSocket);
    setQuickAck(serverSocket);
    return serverSocket;
}

void destroySocket(int serverSocket) {
    shutdown(serverSocket, 2);
    #ifdef _WIN32
    closesocket(serverSocket);
    #else
    close(serverSocket);
    #endif
}

int acceptSocket(int serverSocket) {
    struct sockaddr_in clientAddr;
    socklen_t clientAddrSize = sizeof(clientAddr);
    int clientSocket = ::accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrSize);
    if (clientSocket < 0)
        throw std::runtime_error("Error accepting connection");
    setNoDelay(clientSocket);
    setQuickAck(clientSocket);
    return clientSocket;
}

void initSockets() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        throw std::runtime_error("WSAStartup failed: " + std::to_string(WSAGetLastError()));
    }
#endif
}

void cleanupSockets() {
#ifdef _WIN32
    WSACleanup();
#endif
}

NnConnectionSocketException::NnConnectionSocketException(const std::string message)
    : std::runtime_error(message)
{}

NnTransferSocketException::NnTransferSocketException(int code, const std::string message)
    : code(code), std::runtime_error(message)
{}

NnSocket::NnSocket() {
    this->fd = -1;
}

NnSocket::NnSocket(int fd) : NnSocket() {
    assign(fd);
}

NnSocket::~NnSocket() {
    if (this->fd >= 0)
        destroySocket(this->fd);
}

void NnSocket::assign(int fd) {
    assert(this->fd == -1);
    assert(fd >= 0);
    this->fd = fd;
}

int NnSocket::release() {
    assert(this->fd >= 0);
    int fd = this->fd;
    this->fd = -1;
    return fd;
}

std::unique_ptr<NnNetwork> NnNetwork::serve(int port) {
    NnSocket socketSocket(createServerSocket(port));

    NnUint nSockets;
    NnUint nodeIndex;
    int rootSocketFd = acceptSocket(socketSocket.fd);
    NnSocket rootSocket(rootSocketFd);
    printf("⭕ The root node has connected\n");

    readSocket(rootSocketFd, &nSockets, sizeof(nSockets));
    NnUint nNodes = nSockets - 1; // nSockets - 1 root node
    printf("⭕ nNodes: %d\n", nNodes);
    readSocket(rootSocketFd, &nodeIndex, sizeof(nodeIndex));
    printf("⭕ NodeIndex: %d\n", nodeIndex);

    std::vector<NnSocket> sockets(nSockets);
    sockets[0].assign(rootSocket.release());

    printf("⭕ Socket[0]: accepted root node\n");
    std::vector<std::unique_ptr<char[]>> hosts(nNodes);
    std::vector<int> ports(nNodes);

    NnUint hostLen;
    for (NnUint i = 0; i < nNodes; i++) {
        readSocket(rootSocketFd, &hostLen, sizeof(hostLen));

        std::unique_ptr<char[]> host(new char[hostLen]);
        readSocket(rootSocketFd, host.get(), hostLen);
        hosts[i] = std::move(host);

        readSocket(rootSocketFd, &ports[i], sizeof(ports[i]));
    }

    writeAckPacket(rootSocketFd);

    // We need to wait here until the root node will send a "root is ready" packet
    readAckPacket(rootSocketFd);

    for (NnUint i = 0; i < nNodes; i++) {
        char *host = hosts[i].get();
        int port = ports[i];
        NnUint socketIndex = i + 1;
        if (i >= nodeIndex) {
            printf("⭕ Socket[%d]: connecting to %s:%d worker\n", socketIndex, host, port);
            sockets[socketIndex].assign(connectSocket(host, port));
            printf("⭕ Socket[%d]: connected\n", socketIndex);
        } else {
            printf("⭕ Socket[%d]: wait for %s:%d worker\n", socketIndex, host, port);
            sockets[socketIndex].assign(acceptSocket(socketSocket.fd));
            printf("⭕ Socket[%d]: accepted\n", socketIndex);
        }
    }

    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(&sockets));
}

std::unique_ptr<NnNetwork> NnNetwork::connect(NnUint nSockets, char **hosts, NnUint *ports) {
    assert(nSockets > 0);

    std::vector<NnSocket> sockets(nSockets);
    struct sockaddr_in addr;
    for (NnUint i = 0; i < nSockets; i++) {
        printf("⭕ Socket[%d]: connecting to %s:%d worker\n", i, hosts[i], ports[i]);
        int fd = connectSocket(hosts[i], ports[i]);
        sockets[i].assign(fd);
        writeSocket(fd, &nSockets, sizeof(nSockets));
        writeSocket(fd, &i, sizeof(i)); // send node index
        for (NnUint j = 0; j < nSockets; j++) {
            if (j == i)
                continue;
            NnUint hostLen = strlen(hosts[j]) + 1;
            writeSocket(fd, &hostLen, sizeof(hostLen));
            writeSocket(fd, hosts[j], hostLen);
            writeSocket(fd, &ports[j], sizeof(ports[j]));
        }
        readAckPacket(fd);
        printf("⭕ Socket[%d]: connected\n", i);
    }
    for (NnUint i = 0; i < nSockets; i++) {
        writeAckPacket(sockets[i].fd);
    }
    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(&sockets));
}

NnNetwork::NnNetwork(std::vector<NnSocket> *sockets) {
    this->nSockets = sockets->size();
    this->sockets = new int[nSockets];
    for (NnUint i = 0; i < nSockets; i++)
        this->sockets[i] = sockets->at(i).release();
    this->sentBytes = new NnSize[nSockets];
    this->recvBytes = new NnSize[nSockets];
}

NnNetwork::~NnNetwork() {
    delete[] sentBytes;
    delete[] recvBytes;
    for (NnUint i = 0; i < nSockets; i++)
        destroySocket(sockets[i]);
    delete[] sockets;
    printf("⭕ Network is closed\n");
}

void NnNetwork::setTurbo(bool enabled) {
    for (NnUint i = 0; i < nSockets; i++) {
        ::setNonBlocking(sockets[i], enabled);
    }
}

void NnNetwork::write(const NnUint socketIndex, const void *data, const NnSize size) {
    assert(socketIndex < nSockets);

    NnByte *current = (NnByte *)data;
    int s = sockets[socketIndex];
    for (NnSize chunk = 0; chunk < size; chunk += MAX_CHUNK_SIZE) {
        NnSize chunkSize = chunk + MAX_CHUNK_SIZE < size ? MAX_CHUNK_SIZE : size - chunk;
        writeSocket(s, current, chunkSize);
        current += chunkSize;
    }
    sentBytes[socketIndex] += size;
}

void NnNetwork::read(const NnUint socketIndex, void *data, const NnSize size) {
    assert(socketIndex < nSockets);

    NnByte *current = (NnByte *)data;
    int s = sockets[socketIndex];
    for (NnSize chunk = 0; chunk < size; chunk += MAX_CHUNK_SIZE) {
        NnSize chunkSize = chunk + MAX_CHUNK_SIZE < size ? MAX_CHUNK_SIZE : size - chunk;
        readSocket(s, current, chunkSize);
        current += chunkSize;
    }
    recvBytes[socketIndex] += size;
}

void NnNetwork::writeAck(const NnUint socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    writeAckPacket(sockets[socketIndex]);
}

void NnNetwork::readAck(const NnUint socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    readAckPacket(sockets[socketIndex]);
}

bool NnNetwork::tryReadWithMaxAttempts(NnUint socketIndex, void *data, NnSize size, unsigned long maxAttempts) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    if (tryReadSocket(sockets[socketIndex], data, size, maxAttempts)) {
        recvBytes[socketIndex] += size;
        return true;
    }
    return false;
}

void NnNetwork::writeMany(NnUint n, NnSocketIo *ios) {
    bool isWriting;
    NnSize nBytes = 0;
    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        sentBytes[io->socketIndex] += io->size;
    }
    do {
        isWriting = false;
        for (NnUint i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isWriting = true;
                int socket = sockets[io->socketIndex];
                ssize_t chunkSize = io->size > MAX_CHUNK_SIZE ? MAX_CHUNK_SIZE : io->size;
                ssize_t s = send(socket, (const char*)io->data, chunkSize, 0);
                if (s < 0) {
                    if (isEagainError()) {
                        continue;
                    }
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (s == 0) {
                    throw NnTransferSocketException(0, "Socket closed");
                }
                io->size -= s;
                io->data = (char*)io->data + s;
            }
        }
    } while (isWriting);
}

void NnNetwork::writeAll(void *data, NnSize size) {
    std::vector<NnSocketIo> ios(nSockets);
    for (NnUint i = 0; i < nSockets; i++) {
        NnSocketIo *io = &ios[i];
        io->socketIndex = i;
        io->data = data;
        io->size = size;
    }
    writeMany(nSockets, &ios[0]);
}

void NnNetwork::readMany(NnUint n, NnSocketIo *ios) {
    bool isReading;
    NnSize nBytes = 0;
    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        recvBytes[io->socketIndex] += io->size;
    }
    do {
        isReading = false;
        for (NnUint i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isReading = true;
                int socket = sockets[io->socketIndex];
                ssize_t r = recv(socket, (char*)io->data, io->size, 0);
                if (r < 0) {
                    if (isEagainError()) {
                        continue;
                    }
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (r == 0) {
                    throw NnTransferSocketException(0, "Socket closed");
                }
                io->size -= r;
                io->data = (char*)io->data + r;
            }
        }
    } while (isReading);
}

void NnNetwork::getStats(NnSize *sentBytes, NnSize *recvBytes) {
    *sentBytes = 0;
    *recvBytes = 0;
    for (NnUint i = 0; i < nSockets; i++) {
        *sentBytes += this->sentBytes[i];
        *recvBytes += this->recvBytes[i];
    }
    resetStats();
}

void NnNetwork::resetStats() {
    for (NnUint i = 0; i < nSockets; i++) {
        sentBytes[i] = 0;
        recvBytes[i] = 0;
    }
}

int NnNetwork::getSocketIndexForNode(NnUint targetNodeIndex) const {
// [修复] 针对 1 Root + N Workers 的简单拓扑
    // Root 节点: Node 1 对应 Socket 0
    if (targetNodeIndex > 0) {
        return (int)targetNodeIndex - 1;
    }
    // Worker 节点: Target 0 (Root) 对应 Socket 0
    return 0;
}

void NnNetwork::sendToNode(NnUint targetNodeIndex, const void* data, NnSize size) {
    // 1. 获取对应的 Socket Index
    int socketIndex = getSocketIndexForNode(targetNodeIndex);
    
    // 2. 这里的 socketIndex 就是 sockets 数组的下标
    // write 函数内部会查找 this->sockets[socketIndex]
    write(socketIndex, data, size);
}

void NnNetwork::recvFromNode(NnUint sourceNodeIndex, void* data, NnSize size) {
    int socketIndex = getSocketIndexForNode(sourceNodeIndex);
    read(socketIndex, data, size);
}

static void syncWithRoot(
    NnNetwork *network, 
    NnUint myNodeIndex, 
    NnByte *buffer, 
    NnSize nBytes, 
    NnUint nThreads, 
    NnUint threadIndex,
    const NnStageConfig *stage // [新增] 传入 Stage 信息
) {
    // 1. 确定谁是 Root
    NnUint groupRootIndex = getGroupRootIndex(stage); // 复用之前的辅助函数
    bool amIRoot = (myNodeIndex == groupRootIndex);

    if (amIRoot) {
        // --- Root 发送 (Broadcast) ---
        
        // 确定目标节点列表
        std::vector<int> targetSockets;
        if (stage) {
            // Stage 内广播：只发给组内其他节点
            for(NnUint i=0; i<stage->nNodes; ++i) {
                NnUint target = stage->nodeIndices[i];
                if(target != myNodeIndex) {
                    int sock = network->getSocketIndexForNode(target);
                    if(sock >= 0) targetSockets.push_back(sock);
                }
            }
        } else {
            // 全局广播：发给所有 Socket (简单处理)
            // 注意：这假设 network->nSockets 包含了所有 Worker
            for(NnUint i=0; i<network->nSockets; ++i) targetSockets.push_back(i);
        }

        NnUint nTargets = targetSockets.size();
        if (nTargets == 0) return;

        // 分配给线程
        NnUint nSocketsPerThread = nTargets / nThreads + (nTargets % nThreads > threadIndex ? 1 : 0);
        if (nSocketsPerThread == 0) return;

        NnUint startIdx = 0;
        for (NnUint t = 0; t < threadIndex; ++t) {
            startIdx += nTargets / nThreads + (nTargets % nThreads > t ? 1 : 0);
        }

        std::vector<NnSocketIo> ios(nSocketsPerThread);
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            ios[i].socketIndex = targetSockets[startIdx + i]; // 使用真实的 Socket Index
            ios[i].data = buffer;
            ios[i].size = nBytes;
        }
        network->writeMany(nSocketsPerThread, &ios[0]);

    } else {
        // --- Worker 接收 ---

        if (threadIndex != 0) return; // 接收通常只需要一个线程

        int rootSocketIndex = network->getSocketIndexForNode(groupRootIndex);
        if (rootSocketIndex < 0) {
            // 异常：找不到 Root 的连接
            return; 
        }

        NnSocketIo ios;
        ios.data = buffer;
        ios.size = nBytes;
        ios.socketIndex = rootSocketIndex; // [修正] 使用查找到的 Socket，而不是硬编码 0
        network->readMany(1, &ios);
    }
}

static void syncNodeSlices(
    bool onlyFromWorkerToRoot, 
    NnNetwork *network, 
    NnUint myNodeIndex, 
    NnUint nTotalNodes, 
    NnByte *buffer, 
    NnSize nBytes, 
    NnUint nThreads, 
    NnUint threadIndex, 
    const NnUnevenPartitionPlan *plan,
    const NnStageConfig *stage // 指定同步组
) {
    // ---------------------------------------------------------
    // 0. [核心修改] 确定当前组的 Root 身份
    // ---------------------------------------------------------
    NnUint groupRootIndex = getGroupRootIndex(stage);
    bool amIRoot = (myNodeIndex == groupRootIndex);

    // 1. 确定参与同步的节点列表 (Peers)
    const NnUint* groupNodes = stage ? stage->nodeIndices : nullptr;
    NnUint nGroupNodes = stage ? stage->nNodes : nTotalNodes;

    // 2. 筛选出需要通信的 Socket
    std::vector<int> targetSockets;
    std::vector<NnUint> targetNodeIndices;

    for (NnUint i = 0; i < nGroupNodes; ++i) {
        // 获取目标的全局 ID
        NnUint targetNode = groupNodes ? groupNodes[i] : i;
        
        // 跳过自己
        if (targetNode == myNodeIndex) continue;

        // [修改] 动态的 Root/Worker 判定逻辑
        if (onlyFromWorkerToRoot) {
            // Case A: 我是 Worker (不是本组 Root)
            if (!amIRoot) {
                // Worker 只理会本组的 Root
                if (targetNode != groupRootIndex) continue; 
            }
            // Case B: 我是 Root (本组 Root)
            else { 
                // Root 理会所有人 (接收)，这里不需要 continue，
                // 因为 targetNode 肯定不是我自己(已跳过)，所以是 Worker
            }
        }

        int socketIndex = network->getSocketIndexForNode(targetNode);
        if (socketIndex >= 0) {
            targetSockets.push_back(socketIndex);
            targetNodeIndices.push_back(targetNode);
        }
    }

    // 3. 任务分配给线程 (保持不变)
    NnUint nActiveSockets = targetSockets.size();
    NnUint nSocketsPerThread = nActiveSockets / nThreads + (nActiveSockets % nThreads > threadIndex ? 1 : 0);
    if (nSocketsPerThread == 0) return;

    NnUint startIdx = 0;
    for (NnUint t = 0; t < threadIndex; ++t) {
        startIdx += nActiveSockets / nThreads + (nActiveSockets % nThreads > t ? 1 : 0);
    }

    // 4. 准备切分信息 (Plan Aware) (保持不变)
    std::vector<NnSize> sliceOffsets(nTotalNodes);
    std::vector<NnSize> sliceSizes(nTotalNodes);
    fillUnevenSlices(plan, nTotalNodes, nBytes, sliceOffsets, sliceSizes);

    std::vector<NnSocketIo> ios(nSocketsPerThread);

    // --- 发送阶段 (Send) ---
    bool iShouldSend = true;
    
    // [修改] 如果我是 Root 且模式是 Worker->Root，我不发送
    if (onlyFromWorkerToRoot && amIRoot) iShouldSend = false; 

    if (iShouldSend) {
        //  - 此处展示 TP 组内的 Gather 模式
        // 注意：mySliceData 的偏移量依赖于 fillUnevenSlices 的逻辑
        // 如果使用了之前讨论的“局部偏移重置”，这里 sliceOffsets[myNodeIndex] 也是正确的
        NnByte *mySliceData = &buffer[sliceOffsets[myNodeIndex]];
        NnSize mySliceSize = sliceSizes[myNodeIndex];

        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            NnUint idx = startIdx + i;
            ios[i].socketIndex = targetSockets[idx];
            ios[i].data = mySliceData;
            ios[i].size = mySliceSize;
        }
        network->writeMany(nSocketsPerThread, &ios[0]);
    }

    // --- 接收阶段 (Receive) ---
    bool iShouldRecv = true;
    
    // [修改] 如果我是 Worker 且模式是 Worker->Root，我不接收
    if (onlyFromWorkerToRoot && !amIRoot) iShouldRecv = false; 

    if (iShouldRecv) {
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            NnUint idx = startIdx + i;
            NnUint targetNode = targetNodeIndices[idx];

            ios[i].socketIndex = targetSockets[idx];
            ios[i].data = &buffer[sliceOffsets[targetNode]];
            ios[i].size = sliceSizes[targetNode]; 
        }
        network->readMany(nSocketsPerThread, &ios[0]);
    }
}

static void syncPpSend(NnNetwork *network, NnUint myNodeIndex, NnByte *buffer, NnSize nBytes, 
                       const NnUnevenPartitionPlan *plan) {
    // 1. 找到我所在的 Stage
    const NnStageConfig* myStage = nullptr;
    const NnStageConfig* nextStage = nullptr;
    
    for (NnUint s = 0; s < plan->nStages; ++s) {
        // 检查我是否是该 Stage 的成员
        for (NnUint i = 0; i < plan->stages[s].nNodes; ++i) {
            if (plan->stages[s].nodeIndices[i] == myNodeIndex) {
                myStage = &plan->stages[s];
                // 如果还有下一个 Stage
                if (s + 1 < plan->nStages) {
                    nextStage = &plan->stages[s+1];
                }
                break;
            }
        }
        if (myStage) break;
    }

    // 2. 只有当前 Stage 的 Root 负责发送
    if (myStage && myStage->rootNodeIndex == myNodeIndex) {
        if (nextStage) {
            // 发送给下一阶段的 Root
            // printf("🚀 [PP] Node %u sending %zu bytes to Node %u (Stage %u)\n", 
            //        myNodeIndex, nBytes, nextStage->rootNodeIndex, nextStage->stageIndex);
            
            // 注意：这里需要 network 实现点对点 write
            // 如果网络拓扑不支持直连，可能需要通过 Node 0 中转
            network->sendToNode(nextStage->rootNodeIndex, buffer, nBytes);
        }
    }
}

static void syncPpRecv(NnNetwork *network, NnUint myNodeIndex, NnByte *buffer, NnSize nBytes, 
                       const NnUnevenPartitionPlan *plan) {
    const NnStageConfig* myStage = nullptr;
    const NnStageConfig* prevStage = nullptr;

    for (NnUint s = 0; s < plan->nStages; ++s) {
        for (NnUint i = 0; i < plan->stages[s].nNodes; ++i) {
            if (plan->stages[s].nodeIndices[i] == myNodeIndex) {
                myStage = &plan->stages[s];
                if (s > 0) {
                    prevStage = &plan->stages[s-1];
                }
                break;
            }
        }
        if (myStage) break;
    }

    // 只有当前 Stage 的 Root 负责接收
    if (myStage && myStage->rootNodeIndex == myNodeIndex) {
        if (prevStage) {
            // 从上一阶段的 Root 接收
            // printf("📥 [PP] Node %u receiving %zu bytes from Node %u (Stage %u)\n", 
            //        myNodeIndex, nBytes, prevStage->rootNodeIndex, prevStage->stageIndex);
                   
            network->recvFromNode(prevStage->rootNodeIndex, buffer, nBytes);
        } else {
            // 如果是 Stage 0 的第一层，数据应该来自 Embedding/Input，理论上不走 PP_RECV
            // 除非我们在架构设计上把 Embedding 视为 "Stage -1"
        }
    }
}

NnNetworkNodeSynchronizer::NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, const NnUnevenPartitionPlan *plan) {
    this->network = network;
    this->execution = execution;
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->plan = plan;
    // [新增] 构造时缓存 myStage，避免运行时重复查找
    this->myStage = nullptr;
    if (plan) {
        for (NnUint s = 0; s < plan->nStages; ++s) {
            for (NnUint i = 0; i < plan->stages[s].nNodes; ++i) {
                if (plan->stages[s].nodeIndices[i] == nodeConfig->nodeIndex) {
                    this->myStage = &plan->stages[s];
                    goto stage_found; // 跳出双层循环
                }
            }
        }
    }
stage_found:;
}

void NnNetworkNodeSynchronizer::sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];

    for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
        NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
        NnByte *pipe = execution->pipes[syncConfig->pipeIndex];
        NnPipeConfig *pipeConfig = &netConfig->pipes[syncConfig->pipeIndex];
        NnSize batchBytes = getBytes(pipeConfig->size.floatType, pipeConfig->size.x);

        auto start = std::chrono::high_resolution_clock::now();
        const char* syncTypeStr = "UNKNOWN";

        for (NnUint batchIndex = 0; batchIndex < execution->batchSize; batchIndex++) {
            NnByte *pipeBatch = &pipe[batchIndex * batchBytes];

            if (syncConfig->syncType == SYNC_WITH_ROOT) {
                syncTypeStr = "SYNC_WITH_ROOT";
                syncWithRoot(network, nodeConfig->nodeIndex, pipeBatch, batchBytes, nThreads, threadIndex, this->myStage);
            } else if (syncConfig->syncType == SYNC_NODE_SLICES) {
                syncTypeStr = "SYNC_NODE_SLICES";
                syncNodeSlices(false, network, nodeConfig->nodeIndex, netConfig->nNodes, pipeBatch, batchBytes, nThreads, threadIndex, plan, this->myStage);
            } else if (syncConfig->syncType == SYNC_NODE_SLICES_EXCEPT_ROOT) {
                syncTypeStr = "SYNC_LOGITS";
                syncNodeSlices(true, network, nodeConfig->nodeIndex, netConfig->nNodes, pipeBatch, batchBytes, nThreads, threadIndex, plan, nullptr);
            } 
            else if (syncConfig->syncType == SYNC_PP_SEND) {
                syncTypeStr = "PP_SEND";
                // PP 只要单线程执行一次
                if (threadIndex == 0) {
                    syncPpSend(network, nodeConfig->nodeIndex, pipeBatch, batchBytes, plan);
                }
            }
            else if (syncConfig->syncType == SYNC_PP_RECV) {
                syncTypeStr = "PP_RECV";
                // PP 只要单线程执行一次
                if (threadIndex == 0) {
                    syncPpRecv(network, nodeConfig->nodeIndex, pipeBatch, batchBytes, plan);
                }
            }else {
                throw std::invalid_argument("Unknown sync type");
            }
            if (threadIndex == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
            
            // 阈值过滤：只打印耗时超过 5ms 的操作
            if (elapsedMs > 5.0) {
                printf("⏱️ [Sync Debug] Node %u | Seg %u | %s | Pipe %u | Bytes: %llu | Time: %.2f ms\n", 
                    nodeConfig->nodeIndex, segmentIndex, syncTypeStr, syncConfig->pipeIndex, (unsigned long long)batchBytes, elapsedMs);
            }
        }
        }
    }
}

static void writeString(NnNetwork *network, NnUint socketIndex, char *str) {
    NnUint bytes = std::strlen(str) + 1;
    network->write(socketIndex, &bytes, sizeof(NnUint));
    network->write(socketIndex, str, bytes);
}

static char *readString(NnNetwork *network, NnUint socketIndex) {
    NnUint bytes;
    network->read(socketIndex, &bytes, sizeof(NnUint));
    char *str = new char[bytes];
    network->read(socketIndex, str, bytes);
    return str;
}

NnRootConfigWriter::NnRootConfigWriter(NnNetwork *network) {
    this->network = network;
}

void NnRootConfigWriter::writeNet(NnUint socketIndex, NnNetConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nBatches, sizeof(config->nBatches));
    network->write(socketIndex, &config->nNodes, sizeof(config->nNodes));
    network->write(socketIndex, &config->nPipes, sizeof(config->nPipes));
    for (NnUint pipeIndex = 0; pipeIndex < config->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config->pipes[pipeIndex];
        network->write(socketIndex, &pipeConfig->size, sizeof(pipeConfig->size));
        writeString(network, socketIndex, pipeConfig->name);
    }
    network->write(socketIndex, &config->nPreSyncs, sizeof(config->nPreSyncs));
    for (NnUint preSyncIndex = 0; preSyncIndex < config->nPreSyncs; preSyncIndex++) {
        NnPreSyncConfig *preSyncConfig = &config->preSyncs[preSyncIndex];
        network->write(socketIndex, &preSyncConfig->pipeIndex, sizeof(preSyncConfig->pipeIndex));
    }
    network->readAck(socketIndex);
}

void NnRootConfigWriter::writeNode(NnUint socketIndex, NnNodeConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nodeIndex, sizeof(config->nodeIndex));
    network->write(socketIndex, &config->nBuffers, sizeof(config->nBuffers));
    network->write(socketIndex, &config->nSegments, sizeof(config->nSegments));

    for (NnUint bufferIndex = 0; bufferIndex < config->nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config->buffers[bufferIndex];
        network->write(socketIndex, &bufferConfig->size, sizeof(bufferConfig->size));
        writeString(network, socketIndex, bufferConfig->name);
    }

    for (NnUint segmentIndex = 0; segmentIndex < config->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config->segments[segmentIndex];
        network->write(socketIndex, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->write(socketIndex, &segmentConfig->nOps, sizeof(segmentConfig->nOps));

        for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
            NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
            network->write(socketIndex, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
            network->write(socketIndex, &syncConfig->syncType, sizeof(syncConfig->syncType));
        }
        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            network->write(socketIndex, &opConfig->code, sizeof(opConfig->code));
            network->write(socketIndex, &opConfig->index, sizeof(opConfig->index));
            network->write(socketIndex, &opConfig->weightSize, sizeof(opConfig->weightSize));
            network->write(socketIndex, &opConfig->configSize, sizeof(opConfig->configSize));
            writeString(network, socketIndex, opConfig->name);
            network->write(socketIndex, &opConfig->input, sizeof(opConfig->input));
            network->write(socketIndex, &opConfig->output, sizeof(opConfig->output));
            if (opConfig->configSize > 0)
                network->write(socketIndex, opConfig->config, opConfig->configSize);
        }
    }
    network->readAck(socketIndex);
}

void NnRootConfigWriter::writeToWorkers(NnNetConfig *netConfig, NnNodeConfig *nodeConfigs) {
    for (NnUint nodeIndex = 1; nodeIndex < netConfig->nNodes; nodeIndex++) {
        NnUint socketIndex = nodeIndex - 1;
        writeNet(socketIndex, netConfig);
        writeNode(socketIndex, &nodeConfigs[nodeIndex]);
    }
}

NnWorkerConfigReader::NnWorkerConfigReader(NnNetwork *network) {
    this->network = network;
}

NnNetConfig NnWorkerConfigReader::readNet() {
    network->readAck(ROOT_SOCKET_INDEX);
    NnNetConfig config;
    network->read(ROOT_SOCKET_INDEX, &config.nBatches, sizeof(config.nBatches));
    network->read(ROOT_SOCKET_INDEX, &config.nNodes, sizeof(config.nNodes));
    network->read(ROOT_SOCKET_INDEX, &config.nPipes, sizeof(config.nPipes));
    config.pipes = new NnPipeConfig[config.nPipes];
    for (NnUint pipeIndex = 0; pipeIndex < config.nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config.pipes[pipeIndex];
        network->read(ROOT_SOCKET_INDEX, &pipeConfig->size, sizeof(pipeConfig->size));
        pipeConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }
    network->read(ROOT_SOCKET_INDEX, &config.nPreSyncs, sizeof(config.nPreSyncs));
    config.preSyncs = new NnPreSyncConfig[config.nPreSyncs];
    for (NnUint preSyncIndex = 0; preSyncIndex < config.nPreSyncs; preSyncIndex++) {
        NnPreSyncConfig *preSyncConfig = &config.preSyncs[preSyncIndex];
        network->read(ROOT_SOCKET_INDEX, &preSyncConfig->pipeIndex, sizeof(preSyncConfig->pipeIndex));
    }
    network->writeAck(ROOT_SOCKET_INDEX);
    return config;
}

NnNodeConfig NnWorkerConfigReader::readNode() {
    network->readAck(ROOT_SOCKET_INDEX);

    NnNodeConfig config;
    network->read(ROOT_SOCKET_INDEX, &config.nodeIndex, sizeof(config.nodeIndex));
    network->read(ROOT_SOCKET_INDEX, &config.nBuffers, sizeof(config.nBuffers));
    network->read(ROOT_SOCKET_INDEX, &config.nSegments, sizeof(config.nSegments));

    config.buffers = new NnBufferConfig[config.nBuffers];
    config.segments = new NnSegmentConfig[config.nSegments];

    for (NnUint bufferIndex = 0; bufferIndex < config.nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config.buffers[bufferIndex];
        network->read(ROOT_SOCKET_INDEX, &bufferConfig->size, sizeof(bufferConfig->size));
        bufferConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }

    for (NnUint segmentIndex = 0; segmentIndex < config.nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config.segments[segmentIndex];
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nOps, sizeof(segmentConfig->nOps));

        if (segmentConfig->nSyncs > 0) {
            segmentConfig->syncs = new NnSyncConfig[segmentConfig->nSyncs];

            for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
                NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
                network->read(ROOT_SOCKET_INDEX, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->syncType, sizeof(syncConfig->syncType));
            }
        }

        if (segmentConfig->nOps > 0) {
            segmentConfig->ops = new NnOpConfig[segmentConfig->nOps];

            for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
                NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
                network->read(ROOT_SOCKET_INDEX, &opConfig->code, sizeof(opConfig->code));
                network->read(ROOT_SOCKET_INDEX, &opConfig->index, sizeof(opConfig->index));
                network->read(ROOT_SOCKET_INDEX, &opConfig->weightSize, sizeof(opConfig->weightSize));
                network->read(ROOT_SOCKET_INDEX, &opConfig->configSize, sizeof(opConfig->configSize));
                opConfig->name = readString(network, ROOT_SOCKET_INDEX);
                network->read(ROOT_SOCKET_INDEX, &opConfig->input, sizeof(opConfig->input));
                network->read(ROOT_SOCKET_INDEX, &opConfig->output, sizeof(opConfig->output));
                if (opConfig->configSize > 0) {
                    opConfig->config = new NnByte[opConfig->configSize];
                    network->read(ROOT_SOCKET_INDEX, opConfig->config, opConfig->configSize);
                }
            }
        }
    }
    network->writeAck(ROOT_SOCKET_INDEX);
    return config;
}

NnRootWeightLoader::NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnUint nNodes) {
    this->executor = executor;
    this->network = network;
    this->nNodes = nNodes;
    this->tempSize = 0;
}

NnRootWeightLoader::~NnRootWeightLoader() {
    if (tempSize > 0)
        delete[] temp;
}

void NnRootWeightLoader::finish() {
    NnUint zeroSize = 0;
    for (NnUint socketIndex = 0; socketIndex < nNodes - 1; socketIndex++) {
        network->write(socketIndex, &zeroSize, sizeof(zeroSize));
        network->readAck(socketIndex);
    }
    if (tempSize > 0) {
        delete[] temp;
        tempSize = 0;
    }
}

void NnRootWeightLoader::allocate(NnSize size) {
    if (tempSize < size) {
        if (tempSize > 0)
            delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnRootWeightLoader::writeWeight(NnUint nodeIndex, const char *opName, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    NnUint nameSize = std::strlen(opName) + 1;
    NnUint socketIndex = nodeIndex - 1;
    network->write(socketIndex, &nameSize, sizeof(nameSize));
    network->write(socketIndex, opName, nameSize);
    network->write(socketIndex, &opIndex, sizeof(opIndex));
    network->write(socketIndex, &offset, sizeof(offset));
    network->write(socketIndex, &nBytes, sizeof(nBytes));
    network->write(socketIndex, weight, nBytes);
}

NnSize NnRootWeightLoader::loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    executor->loadWeight(opName, opIndex, 0u, nBytes, weight);
    return nBytes;
}

NnSize NnRootWeightLoader::loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    executor->loadWeight(opName, opIndex, 0u, nBytes, weight);

    if (nNodes > 1u) {
        for (NnUint nodeIndex = 1u; nodeIndex < nNodes; nodeIndex++)
            writeWeight(nodeIndex, opName, opIndex, 0u, nBytes, weight);
    }
    return nBytes;
}

NnSize NnRootWeightLoader::loadRowMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnRowMatmulSlice *slice, NnByte *weight) {
    const NnUint offset = expertIndex * slice->sliceSize.nBytes;
    if (nNodes == 1u) {
        executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
    } else {
        allocate(slice->sliceSize.nBytes);
        for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
            splitRowMatmulWeight(slice, nodeIndex, weight, temp);
            if (nodeIndex == 0u)
                executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, temp);
            else
                writeWeight(nodeIndex, opName, opIndex, offset, slice->sliceSize.nBytes, temp);
        }
    }
    return slice->size.nBytes;
}

NnSize NnRootWeightLoader::loadColMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight) {
    const NnUint offset = expertIndex * slice->sliceSize.nBytes;
    if (nNodes == 1) {
        executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
    } else {
        allocate(slice->sliceSize.nBytes);
        for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
            splitColMatmulWeight(slice, nodeIndex, weight, temp);
            if (nodeIndex == 0)
                executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, temp);
            else
                writeWeight(nodeIndex, opName, opIndex, offset, slice->sliceSize.nBytes, temp);
        }
    }
    return slice->size.nBytes;
}

NnWorkerWeightReader::NnWorkerWeightReader(NnExecutor *executor, NnNetwork *network) {
    this->executor = executor;
    this->network = network;
    this->tempSize = 0;
}

NnWorkerWeightReader::~NnWorkerWeightReader() {
    if (tempSize > 0)
        delete[] temp;
}

void NnWorkerWeightReader::allocate(NnUint size) {
    if (tempSize < size) {
        if (tempSize > 0)
            delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnWorkerWeightReader::read() {
    NnUint nameSize;
    NnUint opIndex;
    NnSize offset;
    NnSize nBytes;
    while (true) {
        network->read(0, &nameSize, sizeof(nameSize));
        if (nameSize == 0) {
            network->writeAck(ROOT_SOCKET_INDEX);
            if (tempSize > 0) {
                delete temp;
                tempSize = 0;
            }
            break;
        }
        std::unique_ptr<char[]> opNamePtr(new char[nameSize]);
        char *opName = opNamePtr.get();
        network->read(ROOT_SOCKET_INDEX, opName, nameSize);
        network->read(ROOT_SOCKET_INDEX, &opIndex, sizeof(opIndex));
        network->read(ROOT_SOCKET_INDEX, &offset, sizeof(offset));
        network->read(ROOT_SOCKET_INDEX, &nBytes, sizeof(nBytes));
        allocate(nBytes);
        network->read(0, temp, nBytes);
        executor->loadWeight(opName, opIndex, offset, nBytes, temp);
        printf("💿 Loaded %22s %3d, %12zu kB\n", opName, opIndex, nBytes / 1024);
    }
    printf("💿 Weights loaded\n");
}
