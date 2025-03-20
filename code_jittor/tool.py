# class tool:

#     def diag(x,diagonal=0):
#         # assert x.ndim==1 or (x.ndim==2 and x.shape[0]==x.shape[1])
#         d = diagonal if diagonal>=0 else -diagonal
#         d_str = f'+{diagonal}' if diagonal>=0 else f'{diagonal}'

#         if x.ndim==1:
#             output_shape = (x.shape[0]+d,)*2
#             return x.reindex(output_shape,[f'i1-{d}' if diagonal>=0 else f'i0-{d}'],overflow_conditions=[f'i0{d_str}!=i1'])
#         else:
#             output_shape = (x.shape[0]-d,)
#             return x.reindex(output_shape,[f'i0+{d}' if diagonal<=0 else 'i0',f'i0+{d}' if diagonal>=0 else 'i0'])
#     def bmm(a, b):
#         # assert len(a.shape) > 2 and len(b.shape) > 2
#         return matmul(a, b)