import socket

def main():
    # Define the server address and port
    server_address = '127.0.0.1'
    server_port = 8080

    # Create a socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Bind the socket to the address and port
        server_socket.bind((server_address, server_port))
        # Listen for incoming connections
        server_socket.listen(1)
        print(f"Server listening on {server_address}:{server_port}")

        # Accept a connection
        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")
            # Receive data from the client
            data = conn.recv(1024)
            if data:
                print(f"Received: {data.decode()}")
                # Send a response back to the client
                conn.sendall(b"Hello, Client!")

if __name__ == "__main__":
    main()
