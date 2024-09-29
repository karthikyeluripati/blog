import  getProjectPostMetadata  from '../../../components/getProjectPostMetadata';
import ProjectPostPreview from '../../../components/PojectPostPreview';

const ProjectPage = () => {
  const postMetadata = getProjectPostMetadata();
  const PostPreviews = postMetadata.map((post)=>(
  <ProjectPostPreview key={post.slug} {...post} />
  ));
  // return <div className='grid grid-cols-1 md:grid-cols-2 gap-4'>{PostPreviews}</div>;
  return (
  <div className=''>
    <h1 className="text-2xl font-bold mb-4">Projects</h1>
    {PostPreviews}
  </div>);
}

export default ProjectPage;