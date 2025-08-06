c = get_config()

# Network and security
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = True
c.ServerApp.allow_root = True

# Authentication
c.ServerApp.token = ''
c.ServerApp.password = ''

# File browser
c.ServerApp.open_browser = False
c.ContentsManager.allow_hidden = True

# Notebook settings
c.NotebookApp.notebook_dir = '/app/notebooks'
c.ServerApp.root_dir = '/app'

# Extensions and plugins
c.LabApp.check_for_updates_class = 'jupyterlab.NeverCheckForUpdate'

# Performance
c.MappingKernelManager.cull_idle_timeout = 3600  # 1 hour
c.MappingKernelManager.cull_interval = 300       # 5 minutes
