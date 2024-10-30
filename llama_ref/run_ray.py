import ray

# Connect to the Ray cluster
ray.init(address='auto')
print("Available resources:", ray.available_resources())


@ray.remote(
  resources={'accelerator_type:TPU-V4': 1.0}
)
def fn():
  # import jax
  # print(jax.devices())
  # return 0
  import run
  run.main()


# Execute the function
result = ray.get([
  fn.remote() for _ in range(4)
])

print("Function result:", result)

