import json

from http.server import BaseHTTPRequestHandler, HTTPServer

from generationlib_helper import LMArgs, init_model_and_state, call_run_generation


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        if isinstance(post_data, (str, bytes)):
            post_data = json.loads(post_data)
        print(f"POST {post_data}")

        model_name = post_data["model_name"]
        prompts = post_data["prompts"]

        # Keep track of failure or success.
        success = True
        error_message = None

        # Extract prompt.
        if success:
            if len(prompts) < 1:
                success = False
                error_message = "Prompt text not provided."

        # TODO: Add support for batch size > 1.
        if success:
            if len(prompts) > 1:
                print("Warning: Multiple prompts provided. Using the first one.")


        # Generate text.
        if success:
            model_response = call_run_generation(model, tokenizer, distributed_state, prompts[0], lm_args)

        # Create a JSON response.
        if success:
            response_data = {
                "form_data": post_data,
                "model_response": model_response,
                "status": "success"
            }
        else:
            response_data = {
                "form_data": post_data,
                "status": error_message
            }

        print(f'SUCCESS {success} {error_message}')

        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode())


def start_server(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    import fire
    import os

    os.environ["PAGER"] = "cat"

    lm_args = fire.Fire(LMArgs)

    print(json.dumps(lm_args.__dict__, indent=4))

    # Init model.
    model, tokenizer, distributed_state = init_model_and_state(lm_args)

    # Start server.
    start_server()
