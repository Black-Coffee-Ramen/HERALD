import re
import os

with open('herald/api/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Startup event
startup_code = """    inspector = inspect(engine)
    
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    try:
        import redis
        from herald.monitoring.redis_queue import DOMAIN_ANALYSIS_QUEUE, VISUAL_ANALYSIS_QUEUE, RedisReliableQueue
        from herald.monitoring.resilience import CircuitBreakerConfig, RedisCircuitBreaker
        redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
        redis_client.ping()
        app.state.redis_client = redis_client
        app.state.domain_queue = RedisReliableQueue(redis_client, DOMAIN_ANALYSIS_QUEUE)
        app.state.visual_queue = RedisReliableQueue(redis_client, VISUAL_ANALYSIS_QUEUE)
        app.state.visual_circuit = RedisCircuitBreaker(redis_client, CircuitBreakerConfig(name="visual_analysis"))
    except Exception as e:
        logger.warning(f"Could not connect to Redis from API layer. {e}")
        app.state.redis_client = None
        app.state.domain_queue = None
        app.state.visual_queue = None
        app.state.visual_circuit = None"""
content = content.replace("    inspector = inspect(engine)", startup_code)

# 2. Remove globals
globals_pattern = re.compile(r'# Redis configuration\nREDIS_HOST = os\.getenv\("REDIS_HOST", "localhost"\).*?raise AttributeError\(f"module \{__name__\} has no attribute \{name\}"\)', re.DOTALL)
deps_code = """def get_redis_client(request: Request):
    return getattr(request.app.state, "redis_client", None)

def get_domain_queue(request: Request):
    return getattr(request.app.state, "domain_queue", None)

def get_visual_queue(request: Request):
    return getattr(request.app.state, "visual_queue", None)

def get_visual_circuit(request: Request):
    return getattr(request.app.state, "visual_circuit", None)"""
content = globals_pattern.sub(deps_code, content)

# 3. trigger_scan
content = content.replace(
    'def trigger_scan(request: Request, scan_req: ScanRequest, current_user: User = Depends(get_current_user)):',
    'def trigger_scan(request: Request, scan_req: ScanRequest, current_user: User = Depends(get_current_user), domain_queue = Depends(get_domain_queue)):')
content = content.replace('    domain_queue = get_domain_queue()\n', '')

# 4. investigate_url
content = content.replace(
    'def investigate_url(request: Request, inv_req: InvestigateRequest, current_user: User = Depends(get_current_user)):',
    'def investigate_url(request: Request, inv_req: InvestigateRequest, current_user: User = Depends(get_current_user), domain_queue = Depends(get_domain_queue)):')
content = content.replace('    domain_queue = get_domain_queue()\n', '')

# 5. prometheus_metrics
content = content.replace(
    'def prometheus_metrics():',
    'def prometheus_metrics(domain_queue = Depends(get_domain_queue), visual_queue = Depends(get_visual_queue), visual_circuit = Depends(get_visual_circuit)):')
content = content.replace('    domain_queue = get_domain_queue()\n', '')
content = content.replace('    visual_queue = get_visual_queue()\n', '')
content = content.replace('    visual_circuit = get_visual_circuit()\n', '')

# 6. get_failed_jobs
content = content.replace(
    'def get_failed_jobs(current_user: User = Depends(get_current_user)):',
    'def get_failed_jobs(current_user: User = Depends(get_current_user), redis_client = Depends(get_redis_client)):')
content = content.replace('    redis_client = get_redis()\n', '')

# 7. retry_failed_jobs
content = content.replace(
    'def retry_failed_jobs(current_user: User = Depends(get_current_user)):',
    'def retry_failed_jobs(current_user: User = Depends(get_current_user), domain_queue = Depends(get_domain_queue)):')
# ensure domain_queue definition is removed if any

# 8. readiness_check
content = content.replace(
    'def readiness_check(db: Session = Depends(get_db)):',
    'def readiness_check(db: Session = Depends(get_db), redis_client = Depends(get_redis_client)):')
content = content.replace('    redis_client = get_redis()\n', '')

# 9. metrics_summary
content = content.replace(
    'def metrics_summary():',
    'def metrics_summary(redis_client = Depends(get_redis_client), domain_queue = Depends(get_domain_queue), visual_queue = Depends(get_visual_queue), visual_circuit = Depends(get_visual_circuit)):')
content = content.replace('    domain_queue = get_domain_queue()\n', '')
content = content.replace('    visual_queue = get_visual_queue()\n', '')
content = content.replace('    redis_client = get_redis()\n', '')
content = content.replace('    visual_circuit = get_visual_circuit()\n', '')

with open('herald/api/main.py', 'w', encoding='utf-8') as f:
    f.write(content)
