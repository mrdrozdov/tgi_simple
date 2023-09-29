"""
# Start server.
python text_generation_inference.py

# Make API calls.
curl "http://0.0.0.0:12345?model_name=gpt2&prompts=first&prompts=last"
> {"url_args":
>     {"model_name": ["gpt2"],
>     "prompts": ["first", "last"]},
>     "prompts": ["first"],
>     "model_response": ["first food. How about flipping some ifs that suddenly popped up? Makes perfect sense to me. I"],
>     "status": "success"
> }
"""

import socket
import json
from urllib.parse import urlparse, parse_qs

from generationlib_helper import LMArgs, init_model_and_state, call_run_generation


def handle_client(client_socket):
    """Handle a client connection."""
    # Receive data from the client
    request = client_socket.recv(1024).decode('utf-8')
    print(f"Received: {request}")

    # Parse the HTTP request to extract the URL
    first_line = request.split("\n")[0]
    url = first_line.split(" ")[1]

    # Extract arguments from the URL
    parsed_url = urlparse(url)
    url_args = parse_qs(parsed_url.query)

    # Keep track of failure or success.
    success = True
    error_message = None

    # Verify model type.
    # NOTE: This check is only included to prevent accidentally using one model when you expect another.
    if success:
        if url_args["model_name"][0] != lm_args.model_name_or_path:
            success = False
            error_message = f"Model name mismatch: {url_args['model_name']} != {lm_args.model_name_or_path}"

    # Extract prompt.
    if success:
        if len(url_args["prompts"]) < 1:
            success = False
            error_message = "Prompt text not provided."

    # TODO: Add support for batch size > 1.
    if success:
        if len(url_args["prompts"]) > 1:
            print("Warning: Multiple prompts provided. Using the first one.")

        prompt_text = url_args["prompts"][0]

    # Generate text.
    if success:
        model_response = call_run_generation(model, tokenizer, distributed_state, prompt_text, lm_args)

    # Create a JSON response
    if success:
        response_data = {
            "url_args": url_args,
            "prompts": [prompt_text],
            "model_response": model_response,
            "status": "success" if success else error_message
        }
    else:
        response_data = {
            "url_args": url_args,
            "status": error_message
        }
    response_json = json.dumps(response_data)

    # Send the JSON response back to the client
    client_socket.sendall(b"HTTP/1.1 200 OK\r\n")
    client_socket.sendall(b"Content-Type: application/json\r\n")
    client_socket.sendall(b"Connection: close\r\n")
    client_socket.sendall(b"\r\n")  # End of headers
    client_socket.sendall(response_json.encode('utf-8'))

    client_socket.close()


def start_server():
    # Define server parameters
    server_ip = "0.0.0.0"
    server_port = 12345

    # Create a socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the IP and port
    server_socket.bind((server_ip, server_port))

    # Start listening with a maximum backlog of 5
    server_socket.listen(5)
    print(f"[*] Listening on {server_ip}:{server_port}")

    try:
        while True:
            # Accept incoming connections
            client_socket, addr = server_socket.accept()
            print(f"[*] Accepted connection from: {addr[0]}:{addr[1]}")

            # Handle the client's request
            handle_client(client_socket)

    except KeyboardInterrupt:
        print("\n[*] Shutting down the server.")
        server_socket.close()


if __name__ == "__main__":
    # Init model.
    lm_args = LMArgs(model_type="gpt2", model_name_or_path="gpt2")
    lm_args.use_cpu = True
    model, tokenizer, distributed_state = init_model_and_state(lm_args)

    # Start server.
    start_server()
