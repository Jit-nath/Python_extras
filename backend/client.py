import socket

def main():
    # Define the server address and port
    server_address = '127.0.0.1'
    server_port = 8080

    # Create a socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to the server
        sock.connect((server_address, server_port))
        print("Connected to the server")

        # Send a message to the server
        message = "Hello, Server!"
        sock.sendall(message.encode())
        print("Message sent to the server")

        # Receive a reply from the server
        reply = sock.recv(1024)
        print("Server reply:", reply.decode())

if __name__ == "__main__":
    main()
