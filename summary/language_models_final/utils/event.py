import asyncio

from utils.aio import wait_first

from payload import ResultTypes


class EventBox:
    def __init__(self, disconnect_event, response_hander):
        self.result_event = asyncio.Event()
        self.disconnect_event = disconnect_event
        self.result = None
        self.result_type = None
        self.response_handler = response_hander

    def _set_result(self, result, result_type):
        self.result = result
        self.result_type = result_type
        self.result_event.set()

    def set_done(self, result):
        self._set_result(result, ResultTypes.DONE)

    def set_error(self, error):
        self._set_result(error, ResultTypes.USER_ERROR)

    def set_application_error(self, error):
        self._set_result(error, ResultTypes.APPLICATION_ERROR)

    def events(self):
        return self.result_event, self.disconnect_event

    def any_event_is_set(self):
        return any(e.is_set() for e in self.events())

    async def wait(self):
        await wait_first([e.wait() for e in self.events()])

    def make_response(self, to_response):
        if self.disconnect_event.is_set():
            self.result_type = ResultTypes.DISCONNECTED
            self.result = None
        return self.response_handler.result_response(self.result_type, self.result, to_response=to_response)
