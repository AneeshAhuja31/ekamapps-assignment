import time
import hashlib
from typing import Dict, Any, Optional, List
from collections import OrderedDict
import json
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()  
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def _generate_key(self, query: str) -> str:
        
        normalized_query = query.lower().strip()
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self._generate_key(query)
        current_time = time.time()
        
        if key in self.cache:
            cached_item = self.cache[key]
            
            if current_time - cached_item['timestamp'] <= self.ttl_seconds:
                self.cache.move_to_end(key)
                self.access_times[key] = current_time
                self.hit_count += 1
                
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_item['data']
            else:
                self._remove_key(key)
        
        self.miss_count += 1
        logger.info(f"Cache miss for query: {query[:50]}...")
        return None
    
    def set(self, query: str, data: Dict[str, Any]) -> None:
        key = self._generate_key(query)
        current_time = time.time()
        
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = {
            'data': data,
            'timestamp': current_time,
            'original_query': query
        }
        self.access_times[key] = current_time
        
        logger.info(f"Cached result for query: {query[:50]}...")
    
    def _remove_key(self, key: str) -> None:
        """Remove a key from cache and access times"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _evict_lru(self) -> None:
        """Remove least recently used item"""
        if self.cache:
            lru_key = next(iter(self.cache))
            self._remove_key(lru_key)
            logger.info("Evicted LRU cache entry")
    
    def clear(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        self._cleanup_expired()
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 2),
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache"""
        current_time = time.time()
        expired_keys = []
        
        for key, cached_item in self.cache.items():
            if current_time - cached_item['timestamp'] > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_key(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of recent cached queries for display"""
        self._cleanup_expired()
        
        recent_queries = []
        
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: self.access_times.get(x[0], 0),
            reverse=True
        )
        
        for key, cached_item in sorted_items[:limit]:
            recent_queries.append({
                "query": cached_item['original_query'],
                "timestamp": cached_item['timestamp'],
                "cached_at": time.strftime(
                    "%Y-%m-%d %H:%M:%S", 
                    time.localtime(cached_item['timestamp'])
                ),
                "query_type": cached_item['data'].get('query_type', 'unknown')
            })
        
        return recent_queries
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern"""
        invalidated_count = 0
        keys_to_remove = []
        
        pattern_lower = pattern.lower()
        
        for key, cached_item in self.cache.items():
            original_query = cached_item.get('original_query', '').lower()
            if pattern_lower in original_query:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._remove_key(key)
            invalidated_count += 1
        
        if invalidated_count > 0:
            logger.info(f"Invalidated {invalidated_count} cache entries matching pattern: {pattern}")
        
        return invalidated_count