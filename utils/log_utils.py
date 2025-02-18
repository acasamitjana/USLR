


class LogBIDSLoader(object):
    def __init__(self, num_files):
        self.num_files = num_files


    def check_length(self, file_list, curr_len=None):
        curr_len = self.num_files if curr_len is None else curr_len

        if len(file_list) != curr_len:
            return {'exit_code': -1, 'file': None,
                    'log': 'Files found: ' + str(len(file_list)) + '; files expected: ' + str(curr_len) +
                           '. Please refine the search.\n' }

        else:
            d = {'exit_code': 0, 'log': ''}
            if curr_len == 1:
                return {**d, 'file': file_list[0]}
            else:
                return {**d, 'file': file_list}
