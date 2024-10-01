import getProjectPostMetadata from '../../../components/getProjectPostMetadata';
import ProjectPostPreview from '../../../components/PojectPostPreview';

const ProjectPage = () => {
  const postMetadata = getProjectPostMetadata();
  
  // Sort the postMetadata array by date in descending order (newest first)
  const sortedPostMetadata = postMetadata.sort((a, b) => {
    const dateA = new Date(a.date.split('-').reverse().join('-'));
    const dateB = new Date(b.date.split('-').reverse().join('-'));
    return dateB - dateA;
  });

  const PostPreviews = sortedPostMetadata.map((post) => (
    <ProjectPostPreview key={post.slug} {...post} />
  ));

  return (
    <div className=''>
      <h1 className="text-2xl font-bold mb-4">Projects</h1>
      {PostPreviews}
    </div>
  );
}

export default ProjectPage;