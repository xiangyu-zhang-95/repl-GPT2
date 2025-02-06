cmp:
	python3  train.py --compile=False --max_iters=10 --self_impl=True > self_impl
	python3  train.py --compile=False --max_iters=10 --self_impl=False > orig_impl
	git diff --no-index self_impl orig_impl

clean:
	rm -rf self_impl orig_impl
