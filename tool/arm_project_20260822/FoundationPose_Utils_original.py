def make_mesh_tensors(mesh, device='cuda', max_tex_size=None):
  mesh_tensors = {}
  if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
    img = np.array(mesh.visual.material.image.convert('RGB'))
    img = img[...,:3]
    if max_tex_size is not None:
      max_size = max(img.shape[0], img.shape[1])
      if max_size>max_tex_size:
        scale = 1/max_size * max_tex_size
        img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
    mesh_tensors['tex'] = torch.as_tensor(img, device=device, dtype=torch.float)[None]/255.0
    mesh_tensors['uv_idx']  = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
    uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
    uv[:,1] = 1 - uv[:,1]
    mesh_tensors['uv']  = uv
  else:
    if mesh.visual.vertex_colors is None:
      logging.info(f"WARN: mesh doesn't have vertex_colors, assigning a pure color")
      mesh.visual.vertex_colors = np.tile(np.array([128,128,128]).reshape(1,3), (len(mesh.vertices), 1))
    mesh_tensors['vertex_color'] = torch.as_tensor(mesh.visual.vertex_colors[...,:3], device=device, dtype=torch.float)/255.0
