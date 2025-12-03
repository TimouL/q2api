#!/usr/bin/env python3
"""
管理 TEMPORARILY_SUSPENDED 账号的脚本
统计和删除 other.api_test.proxy.errors 中包含 TEMPORARILY_SUSPENDED 原因的账号
"""
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from parent directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import init_db, close_db, row_to_dict


def check_suspended_reason(account: dict) -> bool:
    """
    检查账号的 other 字段中是否包含 TEMPORARILY_SUSPENDED 原因
    
    Args:
        account: 账号字典，包含 other 字段
    
    Returns:
        如果包含 TEMPORARILY_SUSPENDED 返回 True，否则返回 False
    """
    try:
        other = account.get('other')
        if not other:
            return False
        
        # 如果 other 是字符串，尝试解析为 JSON
        if isinstance(other, str):
            try:
                other = json.loads(other)
            except json.JSONDecodeError:
                return False
        
        # 检查 api_test.proxy.errors 路径
        api_test = other.get('api_test', {})
        if not isinstance(api_test, dict):
            return False
        
        proxy = api_test.get('proxy', {})
        if not isinstance(proxy, dict):
            return False
        
        errors = proxy.get('errors', [])
        if not isinstance(errors, list):
            return False
        
        # 检查是否有错误消息包含 TEMPORARILY_SUSPENDED
        for error in errors:
            if isinstance(error, str) and 'TEMPORARILY_SUSPENDED' in error:
                return True
        
        return False
    except Exception as e:
        print(f"检查账号时出错: {e}", file=sys.stderr)
        return False


async def list_suspended_accounts() -> list:
    """
    列出所有包含 TEMPORARILY_SUSPENDED 的账号
    
    Returns:
        包含 TEMPORARILY_SUSPENDED 的账号列表
    """
    db = await init_db()
    
    try:
        accounts = await db.fetchall("SELECT * FROM accounts ORDER BY created_at DESC")
        accounts = [row_to_dict(acc) for acc in accounts]
        
        suspended_accounts = []
        for acc in accounts:
            if check_suspended_reason(acc):
                suspended_accounts.append(acc)
        
        return suspended_accounts
    except Exception as e:
        print(f"查询账户时出错: {e}", file=sys.stderr)
        return []


async def show_suspended_stats():
    """显示 TEMPORARILY_SUSPENDED 账号的统计信息"""
    suspended_accounts = await list_suspended_accounts()
    
    if not suspended_accounts:
        print("✅ 没有找到 TEMPORARILY_SUSPENDED 的账号")
        return
    
    print(f"\n⚠️  找到 {len(suspended_accounts)} 个 TEMPORARILY_SUSPENDED 账号:")
    print("-" * 100)
    
    header = "| {:<40s} | {:<20s} | {:<6s} | {:<5s} | {:<5s} | {:<20s} |".format(
        "账号ID", "标签", "启用", "成功", "错误", "创建时间"
    )
    print(header)
    print("-" * 100)
    
    for acc in suspended_accounts:
        acc_id = acc.get('id', '')[:40]  # 截断长ID
        label = (acc.get('label') or '(无)')[:20]
        enabled = "是" if acc.get('enabled') else "否"
        success_count = acc.get('success_count', 0)
        error_count = acc.get('error_count', 0)
        created_at = (acc.get('created_at') or '')[:20]
        
        print("| {:<40s} | {:<20s} | {:<6s} | {:<5d} | {:<5d} | {:<20s} |".format(
            acc_id, label, enabled, success_count, error_count, created_at
        ))
    
    print("-" * 100)
    print(f"\n📊 统计摘要:")
    print(f"   - TEMPORARILY_SUSPENDED 账号总数: {len(suspended_accounts)}")
    enabled_count = sum(1 for acc in suspended_accounts if acc.get('enabled'))
    print(f"   - 其中启用状态: {enabled_count}")
    print(f"   - 其中禁用状态: {len(suspended_accounts) - enabled_count}")
    total_success = sum(acc.get('success_count', 0) for acc in suspended_accounts)
    print(f"   - 总成功调用次数: {total_success}")
    total_errors = sum(acc.get('error_count', 0) for acc in suspended_accounts)
    print(f"   - 总错误次数: {total_errors}")


async def delete_suspended_accounts(confirm: bool = False) -> int:
    """
    删除所有包含 TEMPORARILY_SUSPENDED 的账号
    
    Args:
        confirm: 是否已确认删除
    
    Returns:
        删除的账号数量
    """
    suspended_accounts = await list_suspended_accounts()
    
    if not suspended_accounts:
        print("✅ 没有找到需要删除的 TEMPORARILY_SUSPENDED 账号")
        return 0
    
    if not confirm:
        print(f"\n⚠️  将要删除 {len(suspended_accounts)} 个 TEMPORARILY_SUSPENDED 账号")
        print("请使用 --delete 参数确认删除操作")
        return 0
    
    db = await init_db()
    
    try:
        deleted_count = 0
        for acc in suspended_accounts:
            acc_id = acc.get('id')
            if acc_id:
                await db.execute("DELETE FROM accounts WHERE id = ?", (acc_id,))
                deleted_count += 1
                print(f"已删除账号: {acc_id} ({acc.get('label', '(无)')})")
        
        print(f"\n✅ 成功删除 {deleted_count} 个 TEMPORARILY_SUSPENDED 账号")
        return deleted_count
    except Exception as e:
        print(f"删除账号时出错: {e}", file=sys.stderr)
        return 0


async def main_async():
    """主异步函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='管理 TEMPORARILY_SUSPENDED 账号',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看统计信息
  python manage_suspended_accounts.py
  python manage_suspended_accounts.py --stats
  
  # 删除 TEMPORARILY_SUSPENDED 账号
  python manage_suspended_accounts.py --delete
        """
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='显示 TEMPORARILY_SUSPENDED 账号的统计信息（默认行为）'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='删除所有 TEMPORARILY_SUSPENDED 账号'
    )
    
    args = parser.parse_args()
    
    try:
        if args.delete:
            # 先显示统计
            await show_suspended_stats()
            print("\n" + "=" * 100)
            # 执行删除
            await delete_suspended_accounts(confirm=True)
        else:
            # 默认显示统计
            await show_suspended_stats()
    finally:
        await close_db()


def main():
    """脚本主入口"""
    try:
        asyncio.run(main_async())
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()