
import re


from mydsp.NoiseSource import NoiseSource, NoiseType
from mydsp.Utils import to_number
from utils.MyLogger import MyLogger
from utils.MyLogger import LogLevel
from .dsl_globals import get_context
from .io_commands import noise_params


class WavCommands:


    def cmd_addwaves2(self,args):

        line = ' '.join(args)
        out_symbol = re.search(r'out=(\w+)', line).group(1)
        if out_symbol is None:
            MyLogger.error('WavCommands.cmd_addwaves: no output symbol found')
            return

        src_symbols = re.search(r'srcs=\((.*?)\)', line).group(1).split(',')
        scale = re.search(r"scale=([\d\.]+)",line).group(1)

        if src_symbols is None:
            MyLogger.error('WavCommands.cmd_addwaves: no source symbols found')
            return

        print(out_symbol)
        print(scale)
        print(src_symbols)
        scale = to_number(scale)
        if self.sinks.get(out_symbol) is None:
            MyLogger.error(f'WavCommands.cmd_addwaves: no output found : {out_symbol}')
            return

        for src in src_symbols:
            if self.sources.get(src) is None:
                MyLogger.error(f'WavCommands.cmd_addwaves: source not found {src}')
                return

        src_one = self.sources[src_symbols[0]]
        src_two = self.sources[src_symbols[1]]
        out = self.sinks[out_symbol]

        while True:
            block1 = src_one.getMultiFrame()
            if block1 is None:
                break
            block2 = src_two.getMultiFrame()
            if block2 is not None:
                block1[:, :block1.shape[1]] += scale * block2
            out.writeFrame(block1)

        src_one.close()
        src_two.close()
        out.close()


    def cmd_addwaves(self,args):

        line = ' '.join(args)
        out_symbol = re.search(r'out=(\w+)', line).group(1)
        if out_symbol is None:
            MyLogger.error('WavCommands.cmd_addwaves: no output symbol found')
            return

        src_symbols = re.search(r'srcs=\((.*?)\)', line).group(1).split(',')
        scale = re.search(r"scale=([\d\.]+)",line).group(1)

        if src_symbols is None:
            MyLogger.error('WavCommands.cmd_addwaves: no source symbols found')
            return

        print(out_symbol)
        print(scale)
        print(src_symbols)
        scale = to_number(scale)
        if self.sinks.get(out_symbol) is None:
            MyLogger.error(f'WavCommands.cmd_addwaves: no output found : {out_symbol}')
            return

        srcs = []
        for src in src_symbols:
            if self.sources.get(src) is None:
                MyLogger.error(f'WavCommands.cmd_addwaves: source not found {src}')
                return
            srcs.append(self.sources[src])

        out = self.sinks[out_symbol]

        while True:
            block1 = srcs[0].getMultiFrame()
            if block1 is None:
                break
            for  i in range(1,len(srcs)):
                block = srcs[i].getMultiFrame()
                if block is not None:
                    block1[:, :block1.shape[1]] += block


            out.writeFrame(block1)


        [src.close() for src in srcs]
        out.close()



    def cmd_gennoise(self,args):
        line = ' '.join(args)

        param_str =  line
        print(param_str)
        param_str = param_str.strip("[]")

        params = {}

        pairs = [item.split('=') for item in param_str.split(',') if '=' in item]
        for k, v in pairs:
            params[k] = get_context().vars.get(v, v)

        print(params)
        for param in noise_params:
            print(param)
            if params.get(param,None) is None:
                MyLogger.error(f'WavCommands.cmd_gennoise: param {param} not found')
                return 1

        amplitude = to_number(params['amplitude'])
        num_channels = to_number(params['num_channels'])
        out = params['out']
        frame_size = to_number(params['frame_size'])
        num_frames = to_number(params['num_frames'])
        duration = to_number(params['duration'])

        #nsrc= NoiseSource(NoiseType.WHITE,amplitude,frame_size,num_channels,num_frames)
        print(f"gennoise  out={out} amplitude={amplitude} num_channels={num_channels} frame_size={frame_size} num_frames={num_frames} duration={duration}")
