## Networking: Message Passing API


To facilitate network communication between the simulation server and clients,
we provide a message-passing API that allows clients to interact with the
the server and the simulation. The API is designed to be simple and flexible,

> Note that we also provied read-only shared memory access to the simulation
> state for local processes to access the simulation state without the need
> for network communication. In this case, it is often more efficient to
> access the simulation state directly from memory rather than through the
> message-passing API.


The message-passing API consists of a set of message types that can be sent
between the server and clients. Each message type has a specific format and
purpose, allowing clients to query the simulation state, send commands to the
server, and receive updates from the server. The API is designed to be
asynchronous, allowing clients to send and receive messages independently of
each other. This enables real-time interactions between multiple clients and
the server, making it suitable for multiplayer games, simulations, and other
interactive applications. The API is implemented using a lightweight binary
protocol for efficient communication over the network. The message-passing API
provides the following features:

- **Message Types**: Define a set of message types for querying the simulation
  state, sending commands, and receiving updates.

- **Asynchronous Communication**: Allow clients to send and receive messages
  independently of each other, enabling real-time interactions.

- **Binary Protocol**: Use a lightweight binary protocol for efficient network
  communication. We use `zmq` for message passing.


### Message Types

The message-passing API defines the following message types:

- **Query**: A query message is sent by a client to request information from the
  server. The server responds with a corresponding response message.

- **Command**: A command message is sent by a client to instruct the server to
    perform a specific action. The server may respond with an acknowledgment
    message to confirm that the command was received and executed.

- **Update**: An update message is sent by the server to notify clients of changes
    to the simulation state. This allows clients to stay synchronized with the
    server and receive real-time updates.


### Query

Due to the scale of the simulation, it is often not feasible to send the entire
simulation state in a single message. Instead, clients can send queries to
request specific information from the server. The server responds with a
corresponding response message containing the requested information. Queries
can be used to retrieve information about the simulation state, such as the
state of `BigAtom` objects, a list of `BigAtom` objects in a specific volume,
or the state of the simulation environment.

#### Bounding Volume Big Atom Queries

This is one of the more common queries that clients may send to the server. The
client specifies a bounding volume (e.g., a sphere or axis-aligned bounding box)
and requests a list of `BigAtom` objects that intersect or are contained within
the volume. The server responds with a list of `BigAtom` objects that satisfy
the query.

Since this may involve a large number of `BigAtom` objects, the server may
optimize the query by using spatial indexing structures to quickly find the
relevant `BigAtom` objects. See `spatial-indexing.md`. The server may also apply
filters to the query to limit the number of objects returned or to provide
additional information about the objects.







### Message Passing Interface

#### UUID for Big Atoms

To uniquely identify each big atom and facilitate client-server interactions, we will use UUIDs (Universally Unique Identifiers). This approach allows clients to track identities and manage state efficiently.

#### Message Types and Structures

#### Apply Force

```cpp
struct ApplyForceMessage {
    MessageType type;
    UUID uuid;
    float3 force;
};
```

#### Get Big Atom

```cpp
struct GetBigAtomMessage {
    MessageType type;
    UUID uuid;
};
```

#### Make Big Atom

```cpp
struct MakeBigAtomMessage {
    MessageType type;
    BigAtom newAtom; // or a structure containing initialization parameters
};
```

#### Get Bounding Box

```cpp
struct GetBoundingBoxMessage {
    MessageType type;
    BoundingBox box;
};
```

#### Bounding Box Query

```cpp
struct BoundingBoxQueryMessage {
    MessageType type;
    BoundingBox box;
};
```

#### Mutation Request

```cpp
struct MutationRequestMessage {
    MessageType type;
    UUID uuid;
    float3 force;
};
```

### ZeroMQ Setup and Handling

#### Server-Side Implementation

```cpp
void setupZeroMQ() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    socket.bind("tcp://*:5555");

    while (true) {
        zmq::message_t request;
        socket.recv(&request);

        MessageType msgType = *(MessageType*)request.data();

        switch (msgType) {
            case APPLY_FORCE: {
                ApplyForceMessage* msg = (ApplyForceMessage*)request.data();
                applyForce(msg->uuid, msg->force);
                break;
            }
            case GET_BIG_ATOM: {
                GetBigAtomMessage* msg = (GetBigAtomMessage*)request.data();
                BigAtom atom = getBigAtom(msg->uuid);
                zmq::message_t reply(sizeof(BigAtom));
                memcpy(reply.data(), &atom, sizeof(BigAtom));
                socket.send(reply);
                break;
            }
            case MAKE_BIG_ATOM: {
                MakeBigAtomMessage* msg = (MakeBigAtomMessage*)request.data();
                UUID newUuid = makeBigAtom(msg->newAtom);
                zmq::message_t reply(sizeof(UUID));
                memcpy(reply.data(), &newUuid, sizeof(UUID));
                socket.send(reply);
                break;
            }
            case GET_BOUNDING_BOX: {
                GetBoundingBoxMessage* msg = (GetBoundingBoxMessage*)request.data();
                std::vector<BigAtom> atoms = getBoundingBox(msg->box);
                zmq::message_t reply(atoms.size() * sizeof(BigAtom));
                memcpy(reply.data(), atoms.data(), atoms.size() * sizeof(BigAtom));
                socket.send(reply);
                break;
            }
            case BOUNDING_BOX_QUERY: {
                BoundingBoxQueryMessage* msg = (BoundingBoxQueryMessage*)request.data();
                std::vector<BigAtom> atoms = queryBoundingBox(msg->box);
                zmq::message_t reply(atoms.size() * sizeof(BigAtom));
                memcpy(reply.data(), atoms.data(), atoms.size() * sizeof(BigAtom));
                socket.send(reply);
                break;
            }
            case MUTATION_REQUEST: {
                MutationRequestMessage* msg = (MutationRequestMessage*)request.data();
                mutateBigAtom(msg->uuid, msg->force);
                break;
            }
            default:
                // Handle unknown message type
                break;
        }
    }
}
```

#### Client-Side Implementation

```cpp
void sendApplyForce(zmq::socket_t& socket, UUID uuid, float3 force) {
    ApplyForceMessage msg;
    msg.type = APPLY_FORCE;
    msg.uuid = uuid;
    msg.force = force;
    zmq::message_t request(sizeof(ApplyForceMessage));
    memcpy(request.data(), &msg, sizeof(ApplyForceMessage));
    socket.send(request);
}

BigAtom sendGetBigAtom(zmq::socket_t& socket, UUID uuid) {
    GetBigAtomMessage msg;
    msg.type = GET_BIG_ATOM;
    msg.uuid = uuid;
    zmq::message_t request(sizeof(GetBigAtomMessage));
    memcpy(request.data(), &msg, sizeof(GetBigAtomMessage));
    socket.send(request);

    zmq::message_t reply;
    socket.recv(&reply);
    BigAtom atom = *(BigAtom*)reply.data();
    return atom;
}

UUID sendMakeBigAtom(zmq::socket_t& socket, BigAtom newAtom) {
    MakeBigAtomMessage msg;
    msg.type = MAKE_BIG_ATOM;
    msg.newAtom = newAtom;
    zmq::message_t request(sizeof(MakeBigAtomMessage));
    memcpy(request.data(), &msg, sizeof(MakeBigAtomMessage));
    socket.send(request);

    zmq::message_t reply;
    socket.recv(&reply);
    UUID newUuid = *(UUID*)reply.data();
    return newUuid;
}

std::vector<BigAtom> sendGetBoundingBox(zmq::socket_t& socket, BoundingBox box) {
    GetBoundingBoxMessage msg;
    msg.type = GET_BOUNDING_BOX;
    msg.box = box;
    zmq::message_t request(sizeof(GetBoundingBoxMessage));
    memcpy(request.data(), &msg, sizeof(GetBoundingBoxMessage));
    socket.send(request);

    zmq::message_t reply;
    socket.recv(&reply);
    int numAtoms = reply.size() / sizeof(BigAtom);
    std::vector<BigAtom> atoms(numAtoms);
    memcpy(atoms.data(), reply.data(), reply.size());
    return atoms;
}
```


