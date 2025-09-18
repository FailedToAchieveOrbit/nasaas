"""
Command-line interface for NASaaS
"""

import click
import json
import time
from pathlib import Path
from typing import Dict, Optional

from .client import NASClient, SearchResult
from .core.nas_engine import SearchStatus
from .__init__ import __version__


@click.group()
@click.version_option(version=__version__)
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """NASaaS: Neural Architecture Search as a Service
    
    Autonomous neural network design from natural language descriptions.
    """
    ctx.ensure_object(dict)
    
    # Load configuration
    client_config = {}
    if config:
        with open(config, 'r') as f:
            client_config = json.load(f)
    
    if verbose:
        client_config['log_level'] = 'DEBUG'
    
    # Initialize client
    ctx.obj['client'] = NASClient(config=client_config)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('description')
@click.option('--algorithm', '-a', default='darts', 
              type=click.Choice(['darts', 'enas', 'progressive']),
              help='Search algorithm to use')
@click.option('--max-epochs', type=int, help='Maximum training epochs')
@click.option('--max-latency', type=float, help='Maximum inference latency (ms)')
@click.option('--max-size', type=float, help='Maximum model size (MB)')
@click.option('--target-accuracy', type=float, help='Target accuracy (0-1)')
@click.option('--timeout', default=300, help='Search timeout in seconds')
@click.option('--output', '-o', help='Output file for results')
@click.option('--no-wait', is_flag=True, help='Start search without waiting for completion')
@click.pass_context
def search(ctx, description, algorithm, max_epochs, max_latency, max_size, 
          target_accuracy, timeout, output, no_wait):
    """Start architecture search from natural language description.
    
    DESCRIPTION: Natural language description of the ML task.
    
    Example:
      nasaas search "Classify medical images with high accuracy for mobile devices"
    """
    client = ctx.obj['client']
    verbose = ctx.obj['verbose']
    
    # Build constraints from options
    constraints = {}
    if max_epochs:
        constraints['max_epochs'] = max_epochs
    if max_latency:
        constraints['max_latency_ms'] = max_latency
    if max_size:
        constraints['max_model_size_mb'] = max_size
    if target_accuracy:
        constraints['target_accuracy'] = target_accuracy
    
    # Parse task first if verbose
    if verbose:
        parsed = client.parse_task_description(description)
        click.echo(f"Parsed task type: {parsed['task_type']} (confidence: {parsed['confidence']:.3f})")
        click.echo(f"Performance requirements: {parsed['performance_requirements']}")
        click.echo("")
    
    # Progress callback
    def progress_callback(job_id, status, progress):
        if verbose:
            click.echo(f"Job {job_id}: {status} ({progress:.1f}%)")
        else:
            # Simple progress bar
            bar_length = 30
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            click.echo(f"\r{bar} {progress:.1f}% ({status})", nl=False)
    
    client.add_progress_callback(progress_callback)
    
    try:
        click.echo(f"Starting {algorithm.upper()} architecture search...")
        
        result = client.search_architecture(
            description=description,
            constraints=constraints,
            algorithm=algorithm,
            wait=not no_wait,
            timeout=timeout
        )
        
        if not verbose and not no_wait:
            click.echo()  # New line after progress bar
        
        if result.status == "completed":
            click.echo(click.style("✓ Search completed successfully!", fg='green'))
            
            if result.architecture:
                click.echo(f"Architecture: {result.architecture.get('type', 'Unknown')}")
            
            if result.performance:
                perf = result.performance
                click.echo(f"Performance:")
                if 'accuracy' in perf:
                    click.echo(f"  Accuracy: {perf['accuracy']:.3f}")
                if 'parameters' in perf:
                    click.echo(f"  Parameters: {perf['parameters']:,}")
                if 'latency_ms' in perf:
                    click.echo(f"  Latency: {perf['latency_ms']:.1f}ms")
            
        elif result.status == "failed":
            click.echo(click.style(f"✗ Search failed: {result.error}", fg='red'))
            return
        
        elif no_wait:
            click.echo(f"Search started with job ID: {result.job_id}")
            click.echo(f"Check status with: nasaas status {result.job_id}")
            return
        
        # Save results if requested
        if output:
            output_data = {
                'job_id': result.job_id,
                'description': description,
                'algorithm': algorithm,
                'constraints': constraints,
                'result': {
                    'status': result.status,
                    'architecture': result.architecture,
                    'performance': result.performance,
                    'created_at': result.created_at,
                    'completed_at': result.completed_at
                }
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            click.echo(f"Results saved to: {output}")
        
        # Store job ID for other commands
        ctx.obj['last_job_id'] = result.job_id
        
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red'))
        raise click.Abort()


@cli.command()
@click.argument('job_id', required=False)
@click.pass_context
def status(ctx, job_id):
    """Check status of a search job.
    
    JOB_ID: Job identifier (optional, uses last job if not provided).
    """
    client = ctx.obj['client']
    
    if not job_id:
        job_id = ctx.obj.get('last_job_id')
        if not job_id:
            click.echo("No job ID provided and no previous job found.")
            return
    
    try:
        result = client.get_job_status(job_id)
        
        # Status with color coding
        status_colors = {
            'pending': 'yellow',
            'parsing': 'cyan',
            'loading_data': 'cyan',
            'searching': 'blue',
            'evaluating': 'blue',
            'training_final': 'blue',
            'deploying': 'magenta',
            'completed': 'green',
            'failed': 'red'
        }
        
        color = status_colors.get(result.status, 'white')
        click.echo(f"Job ID: {result.job_id}")
        click.echo(f"Status: {click.style(result.status.upper(), fg=color)}")
        click.echo(f"Progress: {result.progress:.1f}%")
        click.echo(f"Created: {result.created_at}")
        
        if result.completed_at:
            click.echo(f"Completed: {result.completed_at}")
        
        if result.error:
            click.echo(click.style(f"Error: {result.error}", fg='red'))
        
        if result.architecture:
            click.echo(f"\nArchitecture: {result.architecture.get('type', 'Unknown')}")
        
        if result.performance:
            click.echo(f"\nPerformance:")
            for key, value in result.performance.items():
                if isinstance(value, float):
                    click.echo(f"  {key.title()}: {value:.3f}")
                else:
                    click.echo(f"  {key.title()}: {value}")
        
        if result.deployment_info:
            click.echo(f"\nDeployment:")
            click.echo(f"  Platform: {result.deployment_info.get('platform')}")
            click.echo(f"  Endpoint: {result.deployment_info.get('endpoint_url')}")
    
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red'))


@cli.command()
@click.option('--status-filter', type=click.Choice(['pending', 'completed', 'failed']),
              help='Filter jobs by status')
@click.pass_context
def list(ctx, status_filter):
    """List all search jobs."""
    client = ctx.obj['client']
    
    try:
        jobs = client.list_jobs(status=status_filter)
        
        if not jobs:
            click.echo("No jobs found.")
            return
        
        click.echo(f"Found {len(jobs)} job(s):")
        click.echo()
        
        for job in jobs:
            status_color = 'green' if job.status == 'completed' else 'red' if job.status == 'failed' else 'yellow'
            click.echo(f"ID: {job.job_id[:8]}... | "
                      f"Status: {click.style(job.status, fg=status_color)} | "
                      f"Progress: {job.progress:.1f}% | "
                      f"Created: {job.created_at}")
    
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red'))


@cli.command()
@click.argument('job_id', required=False)
@click.option('--platform', '-p', default='local',
              type=click.Choice(['aws', 'gcp', 'azure', 'local']),
              help='Deployment platform')
@click.option('--instance-type', help='Instance type for deployment')
@click.option('--port', type=int, help='Port for local deployment')
@click.pass_context
def deploy(ctx, job_id, platform, instance_type, port):
    """Deploy a completed model.
    
    JOB_ID: Job identifier (optional, uses last job if not provided).
    """
    client = ctx.obj['client']
    
    if not job_id:
        job_id = ctx.obj.get('last_job_id')
        if not job_id:
            click.echo("No job ID provided and no previous job found.")
            return
    
    # Build deployment config
    deploy_config = {}
    if instance_type:
        deploy_config['instance_type'] = instance_type
    if port:
        deploy_config['port'] = port
    
    try:
        click.echo(f"Deploying model to {platform}...")
        
        deployment = client.deploy(
            job_id=job_id,
            platform=platform,
            config=deploy_config
        )
        
        click.echo(click.style("✓ Deployment successful!", fg='green'))
        click.echo(f"Platform: {deployment.get('platform')}")
        click.echo(f"Endpoint: {deployment.get('endpoint_url')}")
        click.echo(f"Status: {deployment.get('status')}")
        
        if 'deployment_id' in deployment:
            click.echo(f"Deployment ID: {deployment['deployment_id']}")
    
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red'))


@cli.command()
@click.argument('description')
@click.pass_context  
def parse(ctx, description):
    """Parse a task description to see how it would be interpreted.
    
    DESCRIPTION: Natural language description of the ML task.
    """
    client = ctx.obj['client']
    
    try:
        parsed = client.parse_task_description(description)
        
        click.echo("Parsed Task Specification:")
        click.echo(f"  Task Type: {parsed['task_type']} (confidence: {parsed['confidence']:.3f})")
        click.echo(f"  Data Type: {parsed['data_type']}")
        
        if parsed['num_classes']:
            click.echo(f"  Classes: {parsed['num_classes']}")
        
        if parsed['input_shape']:
            click.echo(f"  Input Shape: {parsed['input_shape']}")
        
        if parsed['dataset_hints']:
            click.echo(f"  Dataset Hints: {', '.join(parsed['dataset_hints'])}")
        
        if parsed['performance_requirements']:
            click.echo(f"  Performance Requirements:")
            for key, value in parsed['performance_requirements'].items():
                click.echo(f"    {key}: {value}")
        
        if parsed['hardware_constraints']:
            click.echo(f"  Hardware Constraints:")
            for key, value in parsed['hardware_constraints'].items():
                if value:
                    click.echo(f"    {key}: {value}")
    
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red'))


@cli.command()
@click.pass_context
def stats(ctx):
    """Show NAS engine statistics."""
    client = ctx.obj['client']
    
    try:
        statistics = client.get_engine_statistics()
        
        click.echo("NAS Engine Statistics:")
        click.echo(f"  Total Jobs: {statistics.get('total_jobs', 0)}")
        click.echo(f"  Active Jobs: {statistics.get('active_jobs', 0)}")
        click.echo(f"  Completed Jobs: {statistics.get('completed_jobs', 0)}")
        click.echo(f"  Failed Jobs: {statistics.get('failed_jobs', 0)}")
        
        success_rate = statistics.get('success_rate', 0)
        click.echo(f"  Success Rate: {success_rate:.1%}")
        
        avg_accuracy = statistics.get('average_accuracy', 0)
        if avg_accuracy > 0:
            click.echo(f"  Average Accuracy: {avg_accuracy:.3f}")
    
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red'))


@cli.command()
@click.argument('job_id', required=False)
@click.argument('output_file')
@click.pass_context
def save(ctx, job_id, output_file):
    """Save job results to file.
    
    JOB_ID: Job identifier (optional, uses last job if not provided)
    OUTPUT_FILE: Output file path
    """
    client = ctx.obj['client']
    
    if not job_id:
        job_id = ctx.obj.get('last_job_id')
        if not job_id:
            click.echo("No job ID provided and no previous job found.")
            return
    
    try:
        client.save_job_results(job_id, output_file)
        click.echo(f"Job results saved to: {output_file}")
    
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red'))


def main():
    """Entry point for the CLI application."""
    cli()


if __name__ == '__main__':
    main()