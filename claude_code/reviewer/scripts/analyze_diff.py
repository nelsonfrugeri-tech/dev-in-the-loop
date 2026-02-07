#!/usr/bin/env python3
"""
analyze_diff.py - Git Diff Analysis Tool

Analisa diferenças entre branches Git para code review:
- Extrai métricas (linhas, arquivos, complexidade)
- Detecta padrões problemáticos (secrets, N+1, SQL injection)
- Agrupa arquivos em features lógicas
- Gera output estruturado (JSON ou texto)

Usage:
    # Análise de summary
    python analyze_diff.py --base main --compare feature/xyz --format summary
    
    # Análise de arquivo específico
    python analyze_diff.py --file src/api.py --base main --compare feature/xyz
    
    # Análise completa com output JSON
    python analyze_diff.py --base main --compare feature/xyz --format full --output analysis.json
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class FileChange:
    """Representa mudanças em um arquivo."""
    path: str
    status: str  # M=Modified, A=Added, D=Deleted, R=Renamed
    additions: int
    deletions: int
    complexity: str  # low, medium, high, very_high
    
    
@dataclass
class Alert:
    """Representa um alerta detectado automaticamente."""
    type: str
    file: str
    line: int
    severity: str  # critical, high, medium, low
    message: str
    context: Optional[str] = None


@dataclass
class Feature:
    """Representa um agrupamento lógico de arquivos."""
    name: str
    files: List[str]
    impact: str  # low, medium, high
    risk: str  # low, medium, high
    description: str


@dataclass
class AnalysisResult:
    """Resultado completo da análise."""
    stats: Dict
    files: List[FileChange]
    features: List[Feature]
    alerts: List[Alert]
    metrics: Dict


class GitDiffAnalyzer:
    """Analisador de diffs Git."""
    
    # Padrões de detecção
    SECRET_PATTERNS = [
        (r'password\s*=\s*["\']([^"\']+)["\']', 'password hardcoded'),
        (r'api[_-]?key\s*=\s*["\']([^"\']+)["\']', 'api key hardcoded'),
        (r'secret\s*=\s*["\']([^"\']+)["\']', 'secret hardcoded'),
        (r'token\s*=\s*["\']([^"\']+)["\']', 'token hardcoded'),
        (r'aws[_-]?access[_-]?key\s*=\s*["\']([^"\']+)["\']', 'aws key hardcoded'),
    ]
    
    SQL_INJECTION_PATTERNS = [
        (r'execute\s*\(\s*["\'].*%s.*["\'].*%', 'string formatting in SQL'),
        (r'execute\s*\(\s*f["\']', 'f-string in SQL query'),
        (r'execute\s*\(\s*["\'].*\+.*["\']', 'string concatenation in SQL'),
    ]
    
    N_PLUS_ONE_PATTERNS = [
        (r'for\s+\w+\s+in\s+.*:\s*\n.*\.(query|filter|get)\(', 'possible N+1 query in loop'),
    ]
    
    def __init__(self, base: str, compare: str):
        self.base = base
        self.compare = compare
        
    def run_git_command(self, args: List[str]) -> str:
        """Executa comando git e retorna output."""
        try:
            result = subprocess.run(
                ['git'] + args,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar git: {e.stderr}", file=sys.stderr)
            sys.exit(1)
    
    def get_file_stats(self) -> List[FileChange]:
        """Obtém estatísticas de arquivos modificados."""
        # git diff --numstat
        numstat_output = self.run_git_command([
            'diff', '--numstat', f'{self.base}..{self.compare}'
        ])
        
        # git diff --name-status
        status_output = self.run_git_command([
            'diff', '--name-status', f'{self.base}..{self.compare}'
        ])
        
        # Parse status
        status_map = {}
        for line in status_output.strip().split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            status = parts[0][0]  # M, A, D, R
            filepath = parts[-1]
            status_map[filepath] = status
        
        # Parse numstat
        files = []
        for line in numstat_output.strip().split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            additions = int(parts[0]) if parts[0] != '-' else 0
            deletions = int(parts[1]) if parts[1] != '-' else 0
            filepath = parts[2]
            
            # Calcular complexidade
            total_changes = additions + deletions
            if total_changes < 50:
                complexity = 'low'
            elif total_changes < 200:
                complexity = 'medium'
            elif total_changes < 500:
                complexity = 'high'
            else:
                complexity = 'very_high'
            
            files.append(FileChange(
                path=filepath,
                status=status_map.get(filepath, 'M'),
                additions=additions,
                deletions=deletions,
                complexity=complexity
            ))
        
        return files
    
    def get_diff_content(self, filepath: Optional[str] = None) -> str:
        """Obtém conteúdo do diff."""
        cmd = ['diff', f'{self.base}..{self.compare}']
        if filepath:
            cmd.extend(['--', filepath])
        return self.run_git_command(cmd)
    
    def detect_patterns(self, diff_content: str, filepath: str) -> List[Alert]:
        """Detecta padrões problemáticos no diff."""
        alerts = []
        lines = diff_content.split('\n')
        
        for i, line in enumerate(lines):
            # Apenas linhas adicionadas
            if not line.startswith('+'):
                continue
            
            line_content = line[1:]  # Remove '+'
            
            # Detectar secrets
            for pattern, message in self.SECRET_PATTERNS:
                if re.search(pattern, line_content, re.IGNORECASE):
                    alerts.append(Alert(
                        type='secret_hardcoded',
                        file=filepath,
                        line=i + 1,
                        severity='critical',
                        message=f'Possible {message}',
                        context=line_content.strip()[:100]
                    ))
            
            # Detectar SQL injection
            for pattern, message in self.SQL_INJECTION_PATTERNS:
                if re.search(pattern, line_content, re.IGNORECASE):
                    alerts.append(Alert(
                        type='sql_injection',
                        file=filepath,
                        line=i + 1,
                        severity='critical',
                        message=f'Possible SQL injection: {message}',
                        context=line_content.strip()[:100]
                    ))
            
            # Detectar print statements (code smell)
            if re.search(r'\bprint\s*\(', line_content):
                alerts.append(Alert(
                    type='print_statement',
                    file=filepath,
                    line=i + 1,
                    severity='low',
                    message='Print statement found (consider using logger)',
                    context=line_content.strip()[:100]
                ))
            
            # Detectar TODOs
            if re.search(r'\bTODO\b', line_content, re.IGNORECASE):
                alerts.append(Alert(
                    type='todo',
                    file=filepath,
                    line=i + 1,
                    severity='info',
                    message='TODO comment added',
                    context=line_content.strip()[:100]
                ))
        
        # Detectar N+1 (precisa contexto maior)
        full_diff = '\n'.join(lines)
        for pattern, message in self.N_PLUS_ONE_PATTERNS:
            matches = re.finditer(pattern, full_diff, re.MULTILINE)
            for match in matches:
                # Encontrar linha aproximada
                line_num = full_diff[:match.start()].count('\n') + 1
                alerts.append(Alert(
                    type='n_plus_one',
                    file=filepath,
                    line=line_num,
                    severity='high',
                    message=f'Possible N+1 query: {message}',
                    context=match.group(0)[:100]
                ))
        
        return alerts
    
    def calculate_metrics(self, files: List[FileChange], diff_content: str) -> Dict:
        """Calcula métricas do código."""
        python_files = [f for f in files if f.path.endswith('.py')]
        
        # Contar type hints (aproximação)
        type_hints_count = len(re.findall(r':\s*\w+(\[.*?\])?\s*(->\s*\w+)?', diff_content))
        function_defs = len(re.findall(r'^[+\s]*def\s+\w+\s*\(', diff_content, re.MULTILINE))
        type_hints_coverage = type_hints_count / max(function_defs, 1)
        
        # Contar docstrings (aproximação)
        docstrings = len(re.findall(r'^[+\s]*"""', diff_content, re.MULTILINE))
        docstring_coverage = docstrings / max(function_defs, 1)
        
        # Complexidade média
        complexities = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
        avg_complexity = sum(complexities[f.complexity] for f in python_files) / max(len(python_files), 1)
        
        return {
            'type_hints_coverage': round(type_hints_coverage, 2),
            'docstring_coverage': round(docstring_coverage, 2),
            'avg_complexity': round(avg_complexity, 2),
            'total_functions': function_defs,
            'python_files_count': len(python_files)
        }
    
    def group_into_features(self, files: List[FileChange]) -> List[Feature]:
        """Agrupa arquivos em features lógicas."""
        features = []
        
        # Agrupar por diretório
        dir_groups = defaultdict(list)
        for f in files:
            if not f.path.endswith('.py'):
                continue
            parts = Path(f.path).parts
            if len(parts) > 1:
                feature_name = parts[0] if parts[0] != 'src' else (parts[1] if len(parts) > 1 else parts[0])
                dir_groups[feature_name].append(f.path)
        
        # Criar features
        for feature_name, file_list in dir_groups.items():
            # Calcular impacto/risco
            total_changes = sum(
                next((f.additions + f.deletions for f in files if f.path == fp), 0)
                for fp in file_list
            )
            
            has_new_files = any(
                next((f.status == 'A' for f in files if f.path == fp), False)
                for fp in file_list
            )
            
            # Impacto baseado em mudanças
            if total_changes > 300 or has_new_files:
                impact = 'high'
            elif total_changes > 100:
                impact = 'medium'
            else:
                impact = 'low'
            
            # Risco baseado em tipo de arquivo
            has_auth = any('auth' in fp.lower() for fp in file_list)
            has_models = any('model' in fp.lower() or 'schema' in fp.lower() for fp in file_list)
            
            if has_auth:
                risk = 'high'
            elif has_models:
                risk = 'medium'
            else:
                risk = 'low'
            
            features.append(Feature(
                name=feature_name.replace('_', ' ').title(),
                files=file_list,
                impact=impact,
                risk=risk,
                description=f'Changes in {feature_name} module'
            ))
        
        return features
    
    def analyze(self, target_file: Optional[str] = None) -> AnalysisResult:
        """Executa análise completa."""
        # Obter estatísticas de arquivos
        files = self.get_file_stats()
        
        # Filtrar por arquivo específico se solicitado
        if target_file:
            files = [f for f in files if f.path == target_file]
        
        # Obter diff completo
        diff_content = self.get_diff_content(target_file)
        
        # Detectar padrões problemáticos
        alerts = []
        python_files = [f for f in files if f.path.endswith('.py')]
        for file in python_files:
            file_diff = self.get_diff_content(file.path)
            file_alerts = self.detect_patterns(file_diff, file.path)
            alerts.extend(file_alerts)
        
        # Calcular métricas
        metrics = self.calculate_metrics(files, diff_content)
        
        # Agrupar em features
        features = self.group_into_features(files)
        
        # Estatísticas gerais
        total_additions = sum(f.additions for f in files)
        total_deletions = sum(f.deletions for f in files)
        
        stats = {
            'total_files': len(files),
            'python_files': len(python_files),
            'additions': total_additions,
            'deletions': total_deletions,
            'net_change': total_additions - total_deletions
        }
        
        return AnalysisResult(
            stats=stats,
            files=files,
            features=features,
            alerts=alerts,
            metrics=metrics
        )


def format_summary(result: AnalysisResult) -> str:
    """Formata resultado como summary para Opção 1."""
    output = []
    
    output.append("=== Git Diff Analysis Summary ===\n")
    
    # Stats
    output.append("Statistics:")
    output.append(f"  Total files: {result.stats['total_files']}")
    output.append(f"  Python files: {result.stats['python_files']}")
    output.append(f"  Lines added: +{result.stats['additions']}")
    output.append(f"  Lines removed: -{result.stats['deletions']}")
    output.append(f"  Net change: {result.stats['net_change']:+d}")
    output.append("")
    
    # Files by complexity
    output.append("Files by Complexity:")
    by_complexity = defaultdict(list)
    for f in result.files:
        if f.path.endswith('.py'):
            by_complexity[f.complexity].append(f.path)
    
    for complexity in ['very_high', 'high', 'medium', 'low']:
        if files := by_complexity.get(complexity, []):
            output.append(f"  {complexity.replace('_', ' ').title()}: {len(files)} files")
            for fp in files[:3]:  # Mostrar até 3
                output.append(f"    - {fp}")
            if len(files) > 3:
                output.append(f"    ... and {len(files) - 3} more")
    output.append("")
    
    # Features
    if result.features:
        output.append("Features Identified:")
        for feature in result.features:
            output.append(f"  {feature.name}:")
            output.append(f"    Impact: {feature.impact}, Risk: {feature.risk}")
            output.append(f"    Files: {len(feature.files)}")
        output.append("")
    
    # Alerts
    if result.alerts:
        output.append("Alerts:")
        by_severity = defaultdict(list)
        for alert in result.alerts:
            by_severity[alert.severity].append(alert)
        
        for severity in ['critical', 'high', 'medium', 'low', 'info']:
            if alerts := by_severity.get(severity):
                output.append(f"  {severity.upper()}: {len(alerts)} issues")
                for alert in alerts[:3]:
                    output.append(f"    - {alert.file}:{alert.line} - {alert.message}")
                if len(alerts) > 3:
                    output.append(f"    ... and {len(alerts) - 3} more")
        output.append("")
    
    # Metrics
    output.append("Metrics:")
    output.append(f"  Type hints coverage: {result.metrics['type_hints_coverage']:.0%}")
    output.append(f"  Docstring coverage: {result.metrics['docstring_coverage']:.0%}")
    output.append(f"  Average complexity: {result.metrics['avg_complexity']:.1f}/4")
    
    return '\n'.join(output)


def format_json(result: AnalysisResult) -> str:
    """Formata resultado como JSON."""
    data = {
        'stats': result.stats,
        'files': [asdict(f) for f in result.files],
        'features': [asdict(f) for f in result.features],
        'alerts': [asdict(a) for a in result.alerts],
        'metrics': result.metrics
    }
    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze git diff for code review',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--base', required=True, help='Base branch for comparison')
    parser.add_argument('--compare', required=True, help='Compare branch to review')
    parser.add_argument('--file', help='Analyze specific file only')
    parser.add_argument(
        '--format',
        choices=['summary', 'full', 'json'],
        default='summary',
        help='Output format (default: summary)'
    )
    parser.add_argument('--output', help='Save output to file (JSON format)')
    
    args = parser.parse_args()
    
    # Executar análise
    analyzer = GitDiffAnalyzer(args.base, args.compare)
    result = analyzer.analyze(args.file)
    
    # Formatar output
    if args.format == 'json' or args.output:
        output = format_json(result)
    elif args.format == 'full':
        # Full = summary + JSON
        output = format_summary(result)
        output += "\n\n=== JSON Output ===\n"
        output += format_json(result)
    else:
        output = format_summary(result)
    
    # Salvar ou imprimir
    if args.output:
        with open(args.output, 'w') as f:
            f.write(format_json(result))  # Sempre salvar como JSON
        print(f"Analysis saved to: {args.output}")
        if args.format != 'json':
            print("\n" + format_summary(result))
    else:
        print(output)


if __name__ == '__main__':
    main()
