from http.server import HTTPServer, BaseHTTPRequestHandler
import cgi
import json
from PIL import Image
import io
from copy import deepcopy
from .image_processing import compute, load_model_interpreter

ENCODING_FNS = {'threshold': float,
                'overlap': int,
                'iou': float,
                'sizes': str}

class ProcessingRequestBaseHandler(BaseHTTPRequestHandler):

    model_interpreter = None

    def _set_headers(self, data=''):
        code = 200
        self.log_request(code)
        self.send_response_only(code, None)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Date', self.date_time_string())
        self.send_header('Content-Length', len(data))
        self.end_headers()

    def _simple_response(self, message):
        return message.encode("utf8")  # NOTE: must return a bytes object!

    def do_GET(self):
        data = "Not Implemented"
        self._set_headers(data)
        self.wfile.write(self._simple_response(data))

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        image = self._extract_image_from_post()
        if image:
            try:
                compute_params = deepcopy(self.arg_params)
            except AttributeError:
                return_data = self._format_results_dict()
            else:
                detection_params = compute_params.pop('detection_params')
                post_kwargs = self._parse_path_kwargs(self.path.lstrip('/'), path_encoding_fns=ENCODING_FNS)
                detection_params.update(post_kwargs)
                compute_params.update(post_kwargs)

                if not self.__class__.model_interpreter:
                    self.__class__.model_interpreter = load_model_interpreter(compute_params['model'])

                return_data = compute(image=image, interpreter=self.__class__.model_interpreter, detection_params=detection_params, **compute_params)
        else:
            return_data = None
        return_data = self._format_results_dict(return_data)
        self._set_headers(return_data)
        self.wfile.write(return_data)

    def _format_results_dict(self, predictions=None, success=True):
        if not isinstance(predictions, list):
            results_dict = {'success': False, 'predictions': []}
        else:
            results_dict = {'success': success, 'predictions': predictions}
        return self._encode_dict(results_dict)

    def _encode_dict(self, data):
        return json.dumps(data).encode('utf-8')

    def _extract_image_from_post(self):
        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        pdict['CONTENT-LENGTH'] = int(self.headers['Content-Length'])
        if ctype == 'multipart/form-data':
            form = cgi.FieldStorage( fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST', 'CONTENT_TYPE':self.headers['Content-Type']})
            if form.file is not None:
                record_list = [form]
            else:
                record_list = form.list
            try:
                byte_file = io.BytesIO(record_list[0].file.read())
                image = Image.open(byte_file)
            except IOError:
                return None
        return image

    def _parse_path_kwargs(self, path_string, path_encoding_fns={}, path_delim=';'):
        kwargs = {}
        if path_string:
            path_encoding_fns = getattr(self, 'path_encoding_fns', path_encoding_fns)
            path_delim = getattr(self, 'path_delim', path_delim)
            kwarg_pairs = [tuple(k.split('=')) for k in path_string.split(path_delim)]

            for key, value in kwarg_pairs:
                try:
                    kwargs[key] = path_encoding_fns[key](value)
                except:
                    kwargs[key] = value
        return kwargs


class Server:
    def __init__(self, server_class=HTTPServer, handler_base_class=ProcessingRequestBaseHandler):
        self.server_class = server_class
        self.handler_base_class = handler_base_class

    def run(self, addr="localhost", port=8000, arg_params={}):
        handler_class = type('ProcessingRequestHandler', (ProcessingRequestBaseHandler,), {'arg_params': arg_params})
        server_address = (addr, port)
        httpd = self.server_class(server_address, handler_class)

        print(f"Starting httpd server on {addr}:{port}")
        httpd.serve_forever()